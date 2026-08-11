"""Static contract checks for the committed Colab server notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_python_and_keeps_runtime_bounds(self) -> None:
        path = Path("notebooks/medsam2_inference_server_colab.ipynb")
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)

        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        ast.parse(code)

        self.assertIn("userdata.get('NGROK_TOKEN')", code)
        self.assertNotIn("NGROK_TOKEN =", code)
        self.assertIn("'--workers', '1'", code)
        self.assertIn("gpu_gib >= 18", code)
        self.assertIn("major >= 8", code)
        self.assertIn("SAM2_BUILD_CUDA='0'", code)
        self.assertIn("api.colab_server:create_app", code)


if __name__ == "__main__":
    unittest.main()
