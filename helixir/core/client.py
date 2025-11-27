"""HelixDB Client for executing queries and managing connections."""

import logging
from typing import TYPE_CHECKING, Any, Self

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from helixir.core.exceptions import HelixConnectionError, QueryError

if TYPE_CHECKING:
    from helixir.core.config import HelixMemoryConfig

logger = logging.getLogger(__name__)


class HelixDBClient:
    """
    Client for interacting with HelixDB REST API.

    Handles:
    - Connection management
    - Query execution
    - Automatic retries
    - Error handling

    Example:
        >>> client = HelixDBClient(config)
        >>> result = await client.execute_query("getAllUsers", {})
        >>> await client.close()
    """

    def __init__(self, config: HelixMemoryConfig):
        """
        Initialize HelixDB client.

        Args:
            config: HelixMemoryConfig instance
        """

        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._is_connected = False

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers=self._get_headers(),
            )
        return self._client

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def connect(self) -> None:
        """
        Establish connection to HelixDB.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            response = await self.client.post("/getAllUsers", json={})
            if response.status_code not in (200, 404):
                response.raise_for_status()
            self._is_connected = True
            logger.info(
                f"Connected to HelixDB at {self.config.base_url} (instance: {self.config.instance})"
            )
        except httpx.HTTPError as e:
            self._is_connected = False
            raise HelixConnectionError(
                f"Failed to connect to HelixDB at {self.config.base_url}: {e}"
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._is_connected = False
            logger.info("Closed HelixDB connection")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def execute_query(
        self,
        query_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a HelixQL query.

        Args:
            query_name: Name of the query endpoint (e.g., "getAllUsers")
            params: Query parameters as a dictionary

        Returns:
            Query result as a dictionary

        Raises:
            QueryError: If query execution fails
            ConnectionError: If not connected

        Example:
            >>> result = await client.execute_query("getUserByEmail", {"email": "alice@example.com"})
        """
        if not self._is_connected:
            await self.connect()

        if params is None:
            params = {}

        try:
            if "vector_data" in params:
                vector_len = len(params.get("vector_data", []))
                logger.info(f"📊 Query: {query_name} | Vector: {vector_len} dims")
            else:
                logger.debug(
                    f"Executing query: {query_name} with params keys: {list(params.keys())}"
                )

            response = await self.client.post(f"/{query_name}", json=params)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Query result: {result}")

            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Query failed: {query_name}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except Exception:
                error_msg += f" - {e.response.text}"

            if "No value found" in error_msg or "not found" in error_msg.lower():
                logger.debug(error_msg)
            else:
                logger.exception(error_msg)
            raise QueryError(error_msg, query=query_name) from e

        except httpx.HTTPError as e:
            error_msg = f"HTTP error executing query {query_name}: {e}"
            logger.exception(error_msg)
            raise QueryError(error_msg, query=query_name) from e

    async def execute_raw(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """
        Execute a raw HTTP request to HelixDB.

        Args:
            endpoint: API endpoint (e.g., "/getAllUsers")
            data: Request data
            method: HTTP method (GET, POST, etc.)

        Returns:
            Response as a dictionary

        Raises:
            QueryError: If request fails
        """
        if not self._is_connected:
            await self.connect()

        if data is None:
            data = {}

        try:
            if method.upper() == "GET":
                response = await self.client.get(endpoint, params=data)
            elif method.upper() == "POST":
                response = await self.client.post(endpoint, json=data)
            elif method.upper() == "PUT":
                response = await self.client.put(endpoint, json=data)
            elif method.upper() == "DELETE":
                response = await self.client.delete(endpoint, params=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            error_msg = f"HTTP {method} error on {endpoint}: {e}"
            logger.exception(error_msg)
            raise QueryError(error_msg, query=endpoint) from e

    async def health_check(self) -> dict[str, Any]:
        """
        Check HelixDB health status.

        Returns:
            Health status information

        Example:
            >>> health = await client.health_check()
            >>> print(health)
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation of client."""
        status = "connected" if self._is_connected else "disconnected"
        return f"HelixDBClient({self.config.base_url}, instance={self.config.instance}, {status})"
