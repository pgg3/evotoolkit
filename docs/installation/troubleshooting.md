# Troubleshooting

Common issues and quick fixes during installation and setup.

---

## pip cannot find evotoolkit

- Ensure internet access and latest pip: `python -m pip install -U pip`.
- Confirm correct package name: `evotoolkit`.

---

## ImportError: cannot import name 'Solution'

- Check version: `python -c "import evotoolkit, sys; print(evotoolkit.__version__)"`.
- If multiple Python versions, ensure you run the interpreter from the same environment where you installed the package.

---

## CUDA not detected

- Install appropriate CUDA toolkit and drivers for your GPU and OS.
- Some tasks require the `cuda_engineering` extra: `pip install evotoolkit[cuda_engineering]`.

---

## LLM API errors (401/403/SSLError)

- Verify API key is set and valid.
- Check system time and certificate store.
- If using a proxy, ensure it allows HTTPS to the API endpoint.

