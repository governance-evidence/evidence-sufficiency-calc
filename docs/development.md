# Development Workflow

Before opening a change, run the local quality gate:

```bash
make precommit-manual
```

If you are changing Python code, the expected local workflow is:

```bash
make lint
make typecheck
make test
```

CI also builds a wheel and verifies that the packaged DES schema is available
from an installed distribution, not only from an editable checkout.

## Release Workflow

Before cutting a release, run the full quality gate:

```bash
make precommit-manual
```

Then build a wheel locally:

```bash
python -m pip install --upgrade pip wheel
python -m pip wheel --no-deps . -w dist
```

The release contract is not just "the tests pass in editable mode". The wheel
should also be installed into a clean environment and smoke-tested there,
especially for the optional DES adapter and packaged schema resource.
