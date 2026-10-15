# Tests run without network access or credentials.

## Models (mocked)
- test_huggingface_model.py — mocks the transformers pipeline so no model weights are downloaded and no HF token is needed.
- Tests assert graceful behavior when the model returns invalid JSON or fails to load.

Run with: pytest -q