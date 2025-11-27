# Float Validator System

## 🎯 Concept

**Float Validation** - an incremental validation system that allows:

1. **Generate and validate by levels** (0 → 1 → 2 → ...)
2. **Catch problems early** (before deploying to HelixDB)
3. **Beautiful reports** (via rich/click)
4. **Built-in integration** with all framework tools

---

## 🏗️ Architecture

```
float_validator/
├── __init__.py           # Exports
├── validator.py          # FloatValidator - main class
├── reporter.py           # FloatReporter - beautiful reports
├── models.py             # Data models (FloatResult, etc.)
├── hooks.py              # Integration hooks for components
└── README.md             # This documentation
```

---

## 📦 Components

### 1️⃣ **FloatValidator**
```python
class FloatValidator:
    """Main validator with checkpoint tracking."""
    
    def __init__(self, component: str):
        self.component = component
        self.checkpoints: list[FloatCheckpoint] = []
    
    @contextmanager
    def checkpoint(self, name: str) -> FloatCheckpoint:
        """Create validation checkpoint."""
        checkpoint = FloatCheckpoint(
            component=self.component,
            name=name,
            timestamp=datetime.now()
        )
        
        try:
            yield checkpoint
            checkpoint.status = "success"
        except Exception as e:
            checkpoint.status = "failed"
            checkpoint.error = str(e)
        finally:
            self.checkpoints.append(checkpoint)
            self._report(checkpoint)
    
    def get_report(self) -> FloatReport:
        """Get validation report."""
        return FloatReport(checkpoints=self.checkpoints)
```

### 2️⃣ **FloatReporter**
```python
class FloatReporter:
    """Beautiful CLI reports using rich."""
    
    def print_checkpoint(self, checkpoint: FloatCheckpoint):
        """Print single checkpoint result."""
        if checkpoint.status == "success":
            console.print(f"✅ {checkpoint.name}", style="green")
        else:
            console.print(f"❌ {checkpoint.name}", style="red")
    
    def print_report(self, report: FloatReport):
        """Print full validation report."""
        # Beautiful table with rich
        table = Table(title="Float Validation Report")
        table.add_column("Component")
        table.add_column("Checkpoint")
        table.add_column("Status")
        table.add_column("Items")
        # ...
```

### 3️⃣ **Models**
```python
@dataclass
class FloatCheckpoint:
    component: str
    name: str
    timestamp: datetime
    status: str = "pending"
    item_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

@dataclass
class FloatReport:
    checkpoints: list[FloatCheckpoint]
    
    @property
    def is_valid(self) -> bool:
        return all(cp.status == "success" for cp in self.checkpoints)
    
    @property
    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.checkpoints),
            "passed": sum(1 for cp in self.checkpoints if cp.status == "success"),
            "failed": sum(1 for cp in self.checkpoints if cp.status == "failed"),
        }
```

### 4️⃣ **Hooks (Integration)**
```python
class FloatHook:
    """Integration hook for components."""
    
    @staticmethod
    def add_to_query_manager(manager: QueryManager):
        """Add float validation to QueryManager."""
        manager.float_validator = FloatValidator("QueryManager")
        
        # Wrap methods
        original_generate = manager.generate_queries_for_level
        
        def wrapped_generate(level: int):
            with manager.float_validator.checkpoint(f"generate_level_{level}") as ctx:
                queries = original_generate(level)
                ctx.item_count = len(queries)
                ctx.metadata = {"queries": [q.name for q in queries]}
                return queries
        
        manager.generate_queries_for_level = wrapped_generate
```

---

## 🔌 Integration

### **QueryManager**
```python
class QueryManager:
    def __init__(self):
        self.float_validator = FloatValidator("QueryManager")
    
    def generate_queries_for_level(self, level: int):
        with self.float_validator.checkpoint(f"generate_level_{level}") as ctx:
            queries = self._generate(level)
            ctx.record_success(len(queries))
            return queries
```

### **SchemaManager**
```python
class SchemaManager:
    def __init__(self):
        self.float_validator = FloatValidator("SchemaManager")
    
    def register_level(self, level: int, schema: SchemaLevel):
        with self.float_validator.checkpoint(f"register_level_{level}") as ctx:
            self._register(level, schema)
            ctx.record_success(
                item_count=len(schema.nodes) + len(schema.edges)
            )
```

### **HelixDBAdmin**
```python
class HelixDBAdmin:
    def deploy_level(self, level: int, dry_run: bool = True):
        """Deploy level with float validation."""
        
        # 1. Generate queries
        queries = self.query_manager.generate_queries_for_level(level)
        
        # 2. Validate queries
        with self.float_validator.checkpoint(f"validate_level_{level}") as ctx:
            results = [self.validate_query(q.name, level) for q in queries]
            ctx.record_success(len([r for r in results if r.is_valid]))
        
        # 3. Deploy (if not dry_run)
        if not dry_run:
            with self.float_validator.checkpoint(f"deploy_level_{level}") as ctx:
                for query in queries:
                    await client.deploy_query(query.to_helixql())
                ctx.record_success(len(queries))
        
        return self.float_validator.get_report()
```

---

## 🖥️ CLI Integration

```bash
# Validate level with float checkpoints
helix-memory deploy --level 0 --dry-run

# Deploy with float validation
helix-memory deploy --level 0-4 --float

# Show float report
helix-memory float-report

# Export float validation results
helix-memory float-export --output report.json
```

---

## 🎨 Output Example

```
╭─ Float Validation Report ─────────────────────────────────╮
│ Component: QueryManager                                   │
│ ✅ generate_level_0 (2 queries)                           │
│ ✅ generate_level_1 (5 queries)                           │
│ ✅ generate_level_2 (3 queries)                           │
╰───────────────────────────────────────────────────────────╯

╭─ Float Validation Report ─────────────────────────────────╮
│ Component: SchemaManager                                  │
│ ✅ register_level_0 (3 nodes, 2 edges)                    │
│ ✅ register_level_1 (4 nodes, 5 edges)                    │
│ ❌ register_level_2 (Cross-reference error: Memory → Context) │
╰───────────────────────────────────────────────────────────╯

Summary: 5 passed, 1 failed
```

---

## ✅ TODO: Implementation Plan

### Phase 1: Core (High Priority)
- [ ] `models.py`: FloatCheckpoint, FloatReport models
- [ ] `validator.py`: FloatValidator with checkpoint context manager
- [ ] `reporter.py`: FloatReporter with rich output

### Phase 2: Integration
- [ ] `hooks.py`: Integration hooks for all components
- [ ] Add to QueryManager
- [ ] Add to SchemaManager
- [ ] Add to VectorInspector
- [ ] Add to EdgeBuilder

### Phase 3: Admin Methods
- [ ] `generate_queries_for_level(level: int)`
- [ ] `validate_level(level: int)`
- [ ] `deploy_level(level: int, dry_run: bool)`
- [ ] `verify_deployment(level: int)`
- [ ] `export_queries_for_level(level: int, output_path: Path)`

### Phase 4: CLI
- [ ] `helix-memory deploy --dry-run --float`
- [ ] `helix-memory float-report`
- [ ] `helix-memory float-export`

### Phase 5: Makefile
- [ ] `make float-test-level0`
- [ ] `make float-test-all`
- [ ] `make float-report`

---

## 💡 Design Decisions

### Why "Float"?
- Incremental validation = "floating point" (0.0 → 0.5 → 1.0 → ...)
- Each level = checkpoint that "floats up" on success

### Why Context Manager?
```python
with validator.checkpoint("name") as ctx:
    # Auto-start timer
    do_work()
    ctx.record_success()
    # Auto-stop timer, auto-report
```
- Automatic timing
- Automatic exception handling
- Clean API

### Why Rich?
- Beautiful tables
- Colored output
- Progress bars
- Built-in markdown/tables support

---

## 🚀 Usage in Practice

```python
# 1. Setup
admin = HelixDBAdmin(
    query_manager=QueryManager(),
    schema_manager=SchemaManager(),
)

# 2. Validate level (dry-run)
report = admin.deploy_level(level=0, dry_run=True)
print(report.summary)  # {'total': 3, 'passed': 3, 'failed': 0}

# 3. Deploy if valid
if report.is_valid:
    admin.deploy_level(level=0, dry_run=False)

# 4. Continue to next level
report = admin.deploy_level(level=1, dry_run=True)
```

---

## 🔮 Future Enhancements

- **Float history**: Store validation history in JSON/SQLite
- **Float diff**: Compare results between runs
- **Float metrics**: Timing, memory usage, query count
- **Float CI/CD**: Automatic validation in pipeline
- **Float dashboard**: Web UI for viewing reports
