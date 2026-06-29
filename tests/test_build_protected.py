import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_protected.py"


def _load_build_script():
    spec = importlib.util.spec_from_file_location("build_protected", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildProtectedTest(unittest.TestCase):
    def test_clean_output_rejects_repository_root(self):
        module = _load_build_script()
        original_root = module.REPO_ROOT
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            module.REPO_ROOT = tmp_path
            try:
                with self.assertRaises(SystemExit):
                    module.clean_output(tmp_path)
                self.assertTrue(tmp_path.exists())
            finally:
                module.REPO_ROOT = original_root

    def test_check_only_cython_build_does_not_create_output_directory(self):
        module = _load_build_script()
        cython_target = next((ROOT / "src").rglob("*.py"))
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "protected"
            module.compile_cython_targets([cython_target], output, {}, check_only=True)
            self.assertFalse(output.exists())

    def test_compile_cython_requires_explicit_targets(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                module.compile_cython_targets([], Path(tmp), {})

    def test_copy_runtime_files_uses_minimal_allowlist(self):
        module = _load_build_script()
        original_root = module.REPO_ROOT
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            module.REPO_ROOT = repo
            try:
                (repo / "src" / "pkg").mkdir(parents=True)
                (repo / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
                (repo / "src" / "pkg" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
                (repo / "src" / "pkg" / "data.json").write_text("{}", encoding="utf-8")
                (repo / "src" / "pkg" / "README.md").write_text("secret docs", encoding="utf-8")
                (repo / "README.md").write_text("secret docs", encoding="utf-8")
                (repo / "uv.lock").write_text("lock", encoding="utf-8")
                (repo / "Dockerfile").write_text("FROM python", encoding="utf-8")
                (repo / "prisma").mkdir()
                (repo / "prisma" / "schema.prisma").write_text("datasource db {}", encoding="utf-8")
                (repo / "alembic" / "versions").mkdir(parents=True)
                (repo / "alembic" / "script.py.mako").write_text("mako", encoding="utf-8")
                (repo / "alembic" / "versions" / "20260101_0001_init.py").write_text(
                    'revision = "20260101_0001"\n',
                    encoding="utf-8",
                )

                output = repo / "dist-protected" / "app"
                plaintext_runtime_py = module.plaintext_runtime_py_files({})
                copied = module.copy_runtime_files(
                    output,
                    [repo / "src"],
                    list(module.DEFAULT_RUNTIME_INCLUDE_GLOBS),
                    list(module.DEFAULT_RUNTIME_EXCLUDE_GLOBS),
                    plaintext_runtime_py,
                )
                self.assertEqual(
                    sorted(copied),
                    [
                        "alembic/script.py.mako",
                        "alembic/versions/20260101_0001_init.py",
                        "prisma/schema.prisma",
                        "src/pkg/data.json",
                    ],
                )
                self.assertTrue((output / "src" / "pkg" / "data.json").exists())
                self.assertTrue((output / "alembic" / "versions" / "20260101_0001_init.py").exists())
                self.assertFalse((output / "src" / "pkg" / "core.py").exists())
                self.assertFalse((output / "README.md").exists())
                self.assertFalse((output / "src" / "pkg" / "README.md").exists())
                self.assertFalse((output / "uv.lock").exists())
                self.assertFalse((output / "Dockerfile").exists())
            finally:
                module.REPO_ROOT = original_root

    def test_cython_module_name_strips_src_prefix_for_normal_src_layout(self):
        module = _load_build_script()
        target = ROOT / "src" / "pkg" / "core.py"
        pyproject = {
            "tool": {
                "hatch": {
                    "build": {
                        "targets": {
                            "wheel": {
                                "packages": ["src/pkg"],
                            }
                        }
                    }
                }
            }
        }

        self.assertEqual(module.cython_module_name(target, pyproject), "pkg.core")

    def test_cython_module_name_keeps_src_when_src_is_the_package(self):
        module = _load_build_script()
        target = ROOT / "src" / "core.py"
        pyproject = {
            "tool": {
                "hatch": {
                    "build": {
                        "targets": {
                            "wheel": {
                                "packages": ["src"],
                            }
                        }
                    }
                }
            }
        }

        self.assertEqual(module.cython_module_name(target, pyproject), "src.core")

    def test_main_guard_is_rewritten_as_callable_entrypoint(self):
        module = _load_build_script()
        rewritten = module.rewrite_main_guard_as_main_function(
            "\n".join(
                [
                    "events = []",
                    "",
                    "if __name__ == \"__main__\":",
                    "    events.append(\"ran\")",
                    "",
                ]
            )
        )

        namespace = {"__name__": "sample"}
        exec(rewritten, namespace)
        self.assertEqual(namespace["events"], [])

        namespace["main"]()
        self.assertEqual(namespace["events"], ["ran"])

    def test_existing_main_function_is_not_rewritten(self):
        module = _load_build_script()
        source = "def main():\n    return 1\n\nif __name__ == \"__main__\":\n    main()\n"

        self.assertEqual(module.rewrite_main_guard_as_main_function(source), source)

    def test_unlisted_runtime_python_is_rejected(self):
        module = _load_build_script()
        original_root = module.REPO_ROOT
        with TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            module.REPO_ROOT = repo
            try:
                (repo / "src" / "pkg").mkdir(parents=True)
                init_py = repo / "src" / "pkg" / "__init__.py"
                core_py = repo / "src" / "pkg" / "core.py"
                init_py.write_text("", encoding="utf-8")
                core_py.write_text("VALUE = 1\n", encoding="utf-8")

                with self.assertRaises(SystemExit):
                    module.assert_no_unprotected_runtime_python(
                        [repo / "src"],
                        allow_plaintext=False,
                        cython_targets=[core_py],
                        plaintext_runtime_python_files=[],
                    )

                module.assert_no_unprotected_runtime_python(
                    [repo / "src"],
                    allow_plaintext=False,
                    cython_targets=[core_py],
                    plaintext_runtime_python_files=[init_py],
                )
            finally:
                module.REPO_ROOT = original_root

    def test_assert_output_allows_configured_plaintext_runtime_python(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            migration = output / "alembic" / "versions" / "20260101_0001_init.py"
            migration.parent.mkdir(parents=True)
            migration.write_text('revision = "20260101_0001"\n', encoding="utf-8")

            unexpected = output / "src" / "pkg" / "core.py"
            unexpected.parent.mkdir(parents=True)
            unexpected.write_text("VALUE = 1\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                module.assert_output_is_protected(
                    output,
                    ["alembic/versions/20260101_0001_init.py"],
                )

            unexpected.unlink()
            plaintext_files = module.assert_output_is_protected(
                output,
                ["alembic/versions/20260101_0001_init.py"],
            )

        self.assertEqual(plaintext_files, ["alembic/versions/20260101_0001_init.py"])

    def test_forbidden_build_artifacts_are_rejected(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            leaked_c = output / "src" / "pkg" / "core.c"
            leaked_c.parent.mkdir(parents=True)
            leaked_c.write_text("/* generated C */\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                module.assert_no_forbidden_artifacts(output)

            leaked_c.unlink()
            leaked_pyx = output / "src" / "pkg" / "core.pyx"
            leaked_pyx.write_text("# generated pyx\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                module.assert_no_forbidden_artifacts(output)

            leaked_pyx.unlink()
            leaked_pyc = output / "src" / "pkg" / "core.pyc"
            leaked_pyc.write_bytes(b"bytecode")

            with self.assertRaises(SystemExit):
                module.assert_no_forbidden_artifacts(output)

            leaked_pyc.unlink()
            marker = output / "src" / "pkg" / "wrapped.py"
            marker.write_text("__pyarmor__(__name__, __file__, b'')\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                module.assert_no_forbidden_artifacts(output)

    def test_sensitive_audit_rejects_secret_and_respects_allowlist(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = output / "config.toml"
            config.write_text(
                'api_key = "sk-testtoken1234567890abcdef"\n',
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                module.audit_sensitive_artifacts(output, {})

            pyproject = {
                "tool": {
                    "protected-artifacts": {
                        "sensitive-audit-allowlist": ["sk-testtoken"],
                    }
                }
            }
            self.assertEqual(module.audit_sensitive_artifacts(output, pyproject), [])

    def test_sensitive_audit_allows_ordinary_token_field_names(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = output / "config.json"
            config.write_text(
                '{"token": "", "jwt": "", "DATABASE_URL": "${DATABASE_URL}"}\n',
                encoding="utf-8",
            )

            self.assertEqual(module.audit_sensitive_artifacts(output, {}), [])

    def test_sensitive_audit_rejects_quoted_json_secret_keys(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = output / "config.json"
            config.write_text(
                '{"client_secret": "long-secret-value-1234567890"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                module.audit_sensitive_artifacts(output, {})

    def test_sensitive_audit_rejects_generic_high_entropy_token_fields(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = output / "config.toml"
            config.write_text(
                'token = "Abcd1234Efgh5678Ijkl9012"\n'
                'private_key = "abcd1234EFGH5678ijkl9012"\n',
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                module.audit_sensitive_artifacts(output, {})

    def test_sensitive_audit_rejects_weak_password_fallbacks(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = output / "bootstrap.py"
            config.write_text(
                'password = os.getenv("PG_PASSWORD", "postgres")\n',
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                module.audit_sensitive_artifacts(output, {})

    def test_required_strip_fails_when_strip_is_missing(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            with mock.patch.object(module.shutil, "which", return_value=None):
                with self.assertRaises(SystemExit):
                    module.strip_native_extensions(Path(tmp), ["sample.so"], required=True)

    def test_slim_plaintext_runtime_python_is_conservative(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            safe = output / "scripts" / "bootstrap.py"
            safe.parent.mkdir(parents=True)
            safe.write_text(
                '"""module docs"""\n'
                "# setup comment\n"
                "VALUE = 1  # inline comment\n"
                "\n"
                "def run():\n"
                '    """function docs"""\n'
                "    return VALUE\n",
                encoding="utf-8",
            )
            risky = output / "src" / "pkg" / "server.py"
            risky.parent.mkdir(parents=True)
            risky.write_text(
                '"""server docs used by framework tooling"""\n'
                "from fastmcp import FastMCP\n",
                encoding="utf-8",
            )

            slimmed = module.slim_plaintext_runtime_python(
                output,
                ["scripts/bootstrap.py", "src/pkg/server.py"],
            )

            self.assertEqual(slimmed, ["scripts/bootstrap.py"])
            safe_content = safe.read_text(encoding="utf-8")
            self.assertNotIn("module docs", safe_content)
            self.assertNotIn("function docs", safe_content)
            self.assertNotIn("setup comment", safe_content)
            self.assertNotIn("inline comment", safe_content)
            compile(safe_content, str(safe), "exec")
            self.assertIn("server docs used by framework tooling", risky.read_text(encoding="utf-8"))

    def test_remove_manifest_deletes_manifest_only_when_present(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = output / module.MANIFEST_FILENAME
            manifest.write_text("{}", encoding="utf-8")

            self.assertTrue(module.remove_manifest(output))
            self.assertFalse(manifest.exists())
            self.assertFalse(module.remove_manifest(output))

    def test_audit_only_main_does_not_require_runtime_targets(self):
        module = _load_build_script()
        original_root = module.REPO_ROOT
        with TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            output = repo / "dist-protected" / "app"
            output.mkdir(parents=True)
            (output / "config.json").write_text('{"token": ""}\n', encoding="utf-8")
            module.REPO_ROOT = repo
            try:
                with mock.patch.object(module.sys, "argv", ["build_protected.py", "--audit-only"]):
                    self.assertEqual(module.main(), 0)
            finally:
                module.REPO_ROOT = original_root

    def test_manifest_mismatch_is_rejected(self):
        module = _load_build_script()
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            package = output / "src" / "pkg"
            package.mkdir(parents=True)
            init_py = package / "__init__.py"
            init_py.write_text("", encoding="utf-8")
            extension = package / f"core{module.importlib.machinery.EXTENSION_SUFFIXES[0]}"
            extension.write_bytes(b"binary")
            manifest = output / "PROTECTED_ARTIFACT_MANIFEST.json"
            manifest.write_text(
                module.json.dumps({"artifact_files": module.artifact_file_records(output)}),
                encoding="utf-8",
            )
            extension.write_bytes(b"changed")

            original_root = module.REPO_ROOT
            module.REPO_ROOT = output
            try:
                with self.assertRaises(SystemExit):
                    module.verify_existing_output(
                        output,
                        [output / "src" / "pkg" / "core.py"],
                        ["src/pkg/__init__.py"],
                    )
            finally:
                module.REPO_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
