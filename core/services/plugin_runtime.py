"""Plugin runtime singleton.

Core integration layer that wires cleanroom components (EventBus,
PluginOrchestrator, ApprovalStore, EnabledStore, TrustStore, AuditLog,
DiagnosticsCollector, policy evaluation, manifest loading) into a
usable plugin runtime accessible via ``get_plugin_runtime()``.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    BaseRequest,
    PluginRunnerRegistration,
    RunContext,
    RunResult,
    RunnerRegistry,
)

# ── sys.path bootstrap for cleanroom packages ──────────────────────────

_FILE_DIR = Path(__file__).resolve().parent
_ROOT = _FILE_DIR.parent.parent  # backend/
_PROJECT_ROOT = _ROOT.parent  # project root

_PLUGIN_CORE_SRC = _ROOT / "core" / "warehouse" / "lulynx_plugin_core" / "src"
_ROUTE_CONTRACT_SRC = _ROOT / "core" / "warehouse" / "lulynx_route_contract" / "src"

for _p in (_PLUGIN_CORE_SRC, _ROUTE_CONTRACT_SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from lulynx_plugin_core import (
    ApprovalStore,
    AuditLog,
    DiagnosticsCollector,
    EnabledStore,
    EventBus,
    HandlerRegistration,
    PluginDescriptor,
    PluginOrchestrator,
    PluginState,
    TrustStore,
    compute_package_hash,
    evaluate_policy,
    list_capabilities,
    list_hooks,
    load_manifest,
)

from .plugin_loader import PluginLoadError, load_plugin, load_plugin_functions
from .plugin_execution_guard import (
    build_plugin_identity,
    build_plugin_runner_identity,
    build_runner_approval_snapshot,
    collect_sdk_permission_ids,
    elevated_runner_permissions,
    runner_scoped_capabilities,
    runner_required_capabilities,
)
from .plugin_manifest_reader import (
    find_plugin_dir,
    iter_raw_manifest_payloads,
    read_raw_manifest_payload,
)
from .plugin_sdk_job_adapter import submit_sdk_runner_job
from .plugin_sdk_host_policy import build_plugin_sdk_host_policy
from .plugin_sdk_manifest_adapter import collect_sdk_registrations_from_payloads
from .plugin_sdk_runner_executor import execute_plugin_sdk_runner
from .plugin_sdk_registry_adapter import (
    build_sdk_runner_registry_from_registrations,
    build_sdk_status_payload,
)
from .plugin_settings_adapter import (
    build_plugin_settings_panel_payload,
    build_plugin_settings_payload,
    sanitize_plugin_settings,
)
from .plugin_settings_store import PluginSettingsStore
from .plugin_training_snapshot import write_training_hooks_snapshot

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────

_DEFAULT_PLUGIN_ROOT = _PROJECT_ROOT / "plugin"
_DEFAULT_CONFIG_ROOT = _PROJECT_ROOT / "data" / "plugins"


class PluginRuntime:
    """Manages the full plugin lifecycle: scan, load, activate, dispatch.

    Instantiate once via ``get_plugin_runtime()``.
    """

    def __init__(
        self,
        plugin_root: Path | None = None,
        config_root: Path | None = None,
    ) -> None:
        self._plugin_root = plugin_root or _DEFAULT_PLUGIN_ROOT
        self._config_root = config_root or _DEFAULT_CONFIG_ROOT
        self._config_root.mkdir(parents=True, exist_ok=True)

        # Stores
        self._approval = ApprovalStore(self._config_root / "approval.json")
        self._enabled = EnabledStore(self._config_root / "enabled.json")
        self._trust = TrustStore(self._config_root / "trust.json")
        self._audit = AuditLog(self._config_root / "audit.jsonl")
        self._diagnostics = DiagnosticsCollector()

        # Bus + orchestrator
        self._bus = EventBus()
        self._orchestrator = PluginOrchestrator(self._bus)

        # Runtime state
        self._developer_mode = False
        self._loaded_handlers: dict[str, dict[str, Callable]] = {}
        self._manifests: dict[str, PluginManifest] = {}
        self._package_hashes: dict[str, str] = {}
        self._last_results: list[dict[str, Any]] = []
        self._snapshot_path = self._config_root / "training_hooks_snapshot.json"
        self._settings_path = self._config_root / "settings.json"
        self._settings_store = PluginSettingsStore(self._settings_path)
        self._lock = threading.Lock()

    # ── Core reload pipeline ───────────────────────────────────────────

    def reload(self) -> dict[str, Any]:
        """Scan, evaluate, load, and activate all plugins.

        Returns a summary dict with per-plugin status.
        """
        with self._lock:
            return self._reload_unlocked()

    def _reload_unlocked(self) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        self._bus.clear()
        self._loaded_handlers.clear()
        self._manifests.clear()
        self._package_hashes.clear()

        # Step 1: Scan plugin directories
        plugin_dirs: list[Path] = []
        if self._plugin_root.exists():
            for child in sorted(self._plugin_root.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "plugin_manifest.json").exists() or (
                    (child / "manifest.json").exists() and (child / "plugin.py").exists()
                ):
                    plugin_dirs.append(child)

        for plugin_dir in plugin_dirs:
            result = self._load_single_plugin(plugin_dir)
            results.append(result)

        # Step 8: Persist training hooks snapshot
        self._persist_training_snapshot()

        # Step 9: Audit log
        self._audit.append(
            event_type="plugin_reload",
            payload={
                "plugin_count": len(results),
                "active_count": sum(1 for r in results if r.get("active")),
            },
        )

        # Step 10: Return summary
        self._last_results = list(results)
        return {
            "plugin_count": len(results),
            "active_count": sum(1 for r in results if r.get("active")),
            "plugins": results,
        }

    def _load_single_plugin(self, plugin_dir: Path) -> dict[str, Any]:
        """Load a single plugin directory. Returns a result dict."""
        plugin_id = plugin_dir.name
        result: dict[str, Any] = {
            "plugin_id": plugin_id,
            "directory": str(plugin_dir),
            "active": False,
            "error": "",
        }

        # Step 2: Load manifest
        try:
            manifest_path = plugin_dir / "plugin_manifest.json"
            if not manifest_path.exists():
                manifest_path = plugin_dir / "manifest.json"
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            result["error"] = f"manifest_error: {exc}"
            self._audit.append(
                event_type="plugin_load_error",
                plugin_id=plugin_id,
                payload={"error": str(exc)},
            )
            return result

        plugin_id = manifest.plugin_id
        result["plugin_id"] = plugin_id
        result["name"] = manifest.name
        result["version"] = manifest.version
        result["description"] = manifest.description
        result["capabilities"] = list(manifest.capabilities)
        result["settings_panel"] = build_plugin_settings_panel_payload(
            read_raw_manifest_payload(self._plugin_root, plugin_id)
        )

        # Step 3: Compute package hash
        hash_files = list(manifest.signature.files) if manifest.signature else []
        if hash_files:
            pkg_hash, _, missing = compute_package_hash(plugin_dir, hash_files)
            if missing:
                result["error"] = f"missing_files: {missing}"
                return result
        else:
            # Hash all .py files if no explicit file list
            py_files = [str(f.relative_to(plugin_dir)) for f in plugin_dir.rglob("*.py")]
            pkg_hash, _, _ = compute_package_hash(plugin_dir, py_files)

        self._package_hashes[plugin_id] = pkg_hash
        self._manifests[plugin_id] = manifest

        # Step 4: Build identity key and evaluate policy
        signer = manifest.signature.signer if manifest.signature else ""
        identity = build_plugin_identity(plugin_id, manifest.version, pkg_hash, signer)

        approval_result = self._approval.check(
            approval_key=identity,
            required_capabilities=list(manifest.capabilities),
        )
        from lulynx_plugin_core.policy import infer_tier
        tier, _, _ = infer_tier(manifest)
        trust_result = self._trust.evaluate(
            plugin_id=plugin_id,
            version=manifest.version,
            package_hash=pkg_hash,
            signer=signer,
            required_tier=tier,
        )

        enabled_result = self._enabled.resolve(
            plugin_id,
            default_enabled=manifest.enabled_by_default,
        )
        result["enabled"] = bool(enabled_result["enabled"])
        result["enabled_source"] = enabled_result.get("source", "")
        result["enabled_reason"] = enabled_result.get("reason", "")

        policy = evaluate_policy(
            manifest=manifest,
            approval_result=approval_result,
            trust_result=trust_result,
            developer_mode=self._developer_mode,
            activation_enabled=enabled_result["enabled"],
        )

        result["tier"] = policy.required_tier
        result["policy"] = {
            "enabled": policy.enabled,
            "reasons": list(policy.reasons),
            "approved": policy.approved,
            "trust_ok": policy.trust_ok,
        }

        if not policy.enabled:
            result["error"] = f"policy_denied: {', '.join(policy.reasons)}"
            self._audit.append(
                event_type="plugin_policy_denied",
                plugin_id=plugin_id,
                payload={"reasons": list(policy.reasons)},
            )
            return result

        # Step 5: Load plugin code (sandbox)
        if manifest.hooks:
            try:
                handlers = load_plugin(plugin_dir, manifest, permissions=list(manifest.capabilities))
            except PluginLoadError as exc:
                result["error"] = f"load_error: {exc}"
                self._audit.append(
                    event_type="plugin_load_error",
                    plugin_id=plugin_id,
                    payload={"error": str(exc)},
                )
                return result
        else:
            handlers = {}

        self._loaded_handlers[plugin_id] = handlers

        # Step 6: Register handlers on the bus
        for binding in manifest.hooks:
            handler_fn = handlers.get(binding.handler)
            if handler_fn is None:
                continue
            hook_def = None
            from lulynx_plugin_core.hooks import get_hook
            hook_def = get_hook(binding.event)
            self._bus.register(HandlerRegistration(
                plugin_id=plugin_id,
                event=binding.event,
                handler_name=binding.handler,
                handler=handler_fn,
                priority=binding.priority,
                mutable=hook_def.allows_mutation if hook_def else False,
            ))

        # Step 7: Activate in orchestrator
        desc = PluginDescriptor(
            plugin_id=plugin_id,
            display_name=manifest.name,
            version=manifest.version,
            tier=policy.required_tier,
            capabilities=manifest.capabilities,
        )
        self._orchestrator.register(desc)
        self._orchestrator.activate(plugin_id)

        result["active"] = True
        result["loaded"] = True
        result["handlers_loaded"] = list(handlers.keys())

        self._audit.append(
            event_type="plugin_activated",
            plugin_id=plugin_id,
            payload={
                "version": manifest.version,
                "tier": policy.required_tier,
                "handlers": list(handlers.keys()),
            },
        )

        return result

    # ── Management methods ─────────────────────────────────────────────

    def approve_plugin(
        self,
        plugin_id: str,
        approved_by: str = "local-user",
    ) -> dict[str, Any]:
        """Grant approval for a plugin."""
        manifest = self._manifests.get(plugin_id)
        if manifest is None:
            return {"success": False, "error": f"Plugin '{plugin_id}' not loaded"}

        pkg_hash = self._package_hashes.get(plugin_id, "unhashed")
        signer = manifest.signature.signer if manifest.signature else ""
        identity = build_plugin_identity(plugin_id, manifest.version, pkg_hash, signer)

        record = self._approval.grant(
            approval_key=identity,
            plugin_id=plugin_id,
            version=manifest.version,
            package_hash=pkg_hash,
            signer=signer,
            capabilities=runner_required_capabilities(
                manifest=manifest,
                runner_permissions=collect_sdk_permission_ids(self.collect_sdk_registrations(), plugin_id),
            ),
            approved_by=approved_by,
        )
        runner_records = self._grant_plugin_runner_approvals(plugin_id, approved_by=approved_by)
        self._audit.append(
            event_type="plugin_approved",
            plugin_id=plugin_id,
            payload={
                "approved_by": approved_by,
                "approval_key": identity,
                "capabilities": list(record.get("capabilities") or []),
                "runner_approval_count": len(runner_records),
                "runner_approval_keys": [str(item.get("approval_key") or "") for item in runner_records],
            },
        )
        return {"success": True, "record": record, "runner_records": runner_records}

    def approve_plugin_runner(
        self,
        plugin_id: str,
        runner_id: str,
        approved_by: str = "local-user",
    ) -> dict[str, Any]:
        """Grant approval for one SDK runner without approving every runner."""

        registration = self._find_runner_registration(plugin_id, runner_id)
        if registration is None:
            return {"success": False, "error": f"Runner '{runner_id}' for plugin '{plugin_id}' is not declared"}
        record = self._grant_runner_approval(registration, approved_by=approved_by)
        if not record:
            return {"success": False, "error": f"Plugin '{plugin_id}' is not loaded"}
        self._audit.append(
            event_type="plugin_runner_approved",
            plugin_id=plugin_id,
            payload={
                "runner_id": runner_id,
                "approved_by": approved_by,
                "approval_key": str(record.get("approval_key") or ""),
                "capabilities": list(record.get("capabilities") or []),
            },
        )
        return {"success": True, "record": record}

    def revoke_plugin(self, plugin_id: str) -> dict[str, Any]:
        """Revoke approval for a plugin."""
        removed = self._approval.revoke(plugin_id)
        self._audit.append(
            event_type="plugin_revoked",
            plugin_id=plugin_id,
            payload={"records_removed": removed},
        )
        return {"success": True, "records_removed": removed}

    def set_plugin_enabled(
        self,
        plugin_id: str,
        enabled: bool,
        updated_by: str = "local-user",
    ) -> dict[str, Any]:
        """Set an enabled/disabled override for a plugin."""
        record = self._enabled.set_override(
            plugin_id, enabled=enabled, updated_by=updated_by,
        )
        self._audit.append(
            event_type="plugin_enabled_changed",
            plugin_id=plugin_id,
            payload={"enabled": enabled, "updated_by": updated_by},
        )
        return {"success": True, "record": record}

    def reset_plugin_enabled(self, plugin_id: str) -> dict[str, Any]:
        """Clear the enabled override for a plugin."""
        removed = self._enabled.clear_override(plugin_id)
        self._audit.append(
            event_type="plugin_enabled_reset",
            plugin_id=plugin_id,
            payload={"records_removed": removed},
        )
        return {"success": True, "records_removed": removed}

    def toggle_developer_mode(self, enabled: bool) -> dict[str, Any]:
        """Enable or disable developer mode (bypasses approval/trust)."""
        self._developer_mode = bool(enabled)
        self._audit.append(
            event_type="developer_mode_toggled",
            payload={"enabled": self._developer_mode},
        )
        return {"success": True, "developer_mode": self._developer_mode}

    # ── Event dispatch ─────────────────────────────────────────────────

    def emit_event(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch an event through the bus and record diagnostics."""
        report = self._bus.emit(event, payload)
        self._diagnostics.record(report)
        return report

    def emit_mutation_event(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a mutation event (modify_loss etc.) and record diagnostics."""
        report = self._bus.emit(event, payload, capture_result=True)
        self._diagnostics.record(report)
        return report

    # ── Queries ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of the runtime state."""
        orch_snap = self._orchestrator.snapshot()
        plugins = list(self._last_results)
        plugin_count = len(plugins) if plugins else int(orch_snap.get("plugin_count") or 0)
        active_count = sum(1 for p in plugins if p.get("active")) if plugins else int(orch_snap.get("active_count") or 0)
        return {
            "developer_mode": self._developer_mode,
            "plugin_root": str(self._plugin_root),
            "config_root": str(self._config_root),
            "plugin_count": plugin_count,
            "active_count": active_count,
            "plugins": plugins,
            "orchestrator": orch_snap,
            "diagnostics": self._diagnostics.summary(),
        }

    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def backend_root(self) -> Path:
        return _ROOT

    def get_manifests(self) -> dict[str, dict[str, Any]]:
        """Return loaded manifests as plain dicts."""
        return {
            pid: {
                "plugin_id": m.plugin_id,
                "name": m.name,
                "version": m.version,
                "entry": m.entry,
                "description": m.description,
                "capabilities": list(m.capabilities),
                "hooks": [
                    {"event": h.event, "handler": h.handler, "priority": h.priority}
                    for h in m.hooks
                ],
            }
            for pid, m in self._manifests.items()
        }

    def get_diagnostics_summary(self) -> dict[str, Any]:
        """Return diagnostics summary."""
        return self._diagnostics.summary()

    def get_audit_log(self, limit: int = 200) -> list[dict]:
        """Return recent audit log entries."""
        return self._audit.recent(limit)

    def _read_settings_all(self) -> dict[str, Any]:
        return self._settings_store.read_all()

    def _write_settings_all(self, data: dict[str, Any]) -> None:
        self._settings_store.write_all(data)

    def _read_raw_manifest_payload(self, plugin_id: str) -> dict[str, Any]:
        return read_raw_manifest_payload(self._plugin_root, plugin_id)

    def _iter_raw_manifest_payloads(self) -> list[dict[str, Any]]:
        return iter_raw_manifest_payloads(self._plugin_root)

    def collect_sdk_registrations(self) -> dict[str, Any]:
        """Collect request-native plugin declarations without executing code."""
        return collect_sdk_registrations_from_payloads(self._iter_raw_manifest_payloads())

    def build_sdk_runner_registry(self) -> RunnerRegistry:
        """Build a discovery-only registry for declarative plugin runners."""

        return build_sdk_runner_registry_from_registrations(self.collect_sdk_registrations())

    def build_sdk_execution_registry(self) -> RunnerRegistry:
        """Build a permissioned registry for loaded plugin runner execution."""

        return build_sdk_runner_registry_from_registrations(
            self.collect_sdk_registrations(),
            runtime=self,
            execution_enabled=True,
        )

    def sdk_status(self) -> dict[str, Any]:
        """Return SDK declarations and runner capabilities for UI approval flows."""

        registrations = self.collect_sdk_registrations()
        return build_sdk_status_payload(
            registrations=registrations,
            discovery_capabilities=self.build_sdk_runner_registry().capabilities(),
            execution_capabilities=self.build_sdk_execution_registry().capabilities(),
            approval_snapshot_builder=self._runner_approval_snapshot,
            find_plugin_dir=self._find_plugin_dir,
            sandbox_context_metadata=self._sdk_host_policy_metadata(),
        )

    def execute_sdk_runner_job(
        self,
        *,
        runner_id: str,
        payload: dict[str, Any] | None,
        project_root: Path,
        backend_root: Path | None,
        job_manager: Any,
        requested_by: str = "ui-user",
    ) -> dict[str, Any]:
        """Submit a plugin SDK runner through the common job manager."""

        return submit_sdk_runner_job(
            registry=self.build_sdk_execution_registry(),
            runner_id=runner_id,
            payload=payload,
            project_root=project_root,
            backend_root=backend_root,
            job_manager=job_manager,
            approval_snapshot_builder=self._runner_approval_snapshot,
            host_policy_metadata=self._sdk_host_policy_metadata(),
            requested_by=requested_by,
        )

    def _approved_runner_permissions_for(self, plugin_id: str, permissions: list[str]) -> list[str]:
        registration = PluginRunnerRegistration(
            plugin_id=plugin_id,
            runner_id="approval-probe",
            request_schema_id="approval.probe",
            entrypoint="plugin.py:probe",
            permissions=list(permissions or []),
        )
        return self._approved_runner_permissions(registration)

    def _grant_plugin_runner_approvals(self, plugin_id: str, *, approved_by: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in self.collect_sdk_registrations().get("runners", []) or []:
            if not isinstance(item, dict) or str(item.get("plugin_id") or "").strip() != plugin_id:
                continue
            try:
                registration = PluginRunnerRegistration.model_validate(item)
            except Exception:
                continue
            record = self._grant_runner_approval(registration, approved_by=approved_by)
            if record:
                records.append(record)
        return records

    def _grant_runner_approval(self, registration: PluginRunnerRegistration, *, approved_by: str) -> dict[str, Any]:
        manifest = self._manifests.get(registration.plugin_id)
        package_hash = self._package_hashes.get(registration.plugin_id, "")
        if manifest is None or not package_hash:
            return {}
        signer = manifest.signature.signer if manifest.signature else ""
        approval_key = build_plugin_runner_identity(
            registration.plugin_id,
            manifest.version,
            package_hash,
            signer,
            runner_id=registration.runner_id,
            request_schema_id=registration.request_schema_id,
            entrypoint=registration.entrypoint,
        )
        return self._approval.grant(
            approval_key=approval_key,
            plugin_id=registration.plugin_id,
            version=manifest.version,
            package_hash=package_hash,
            signer=signer,
            capabilities=runner_scoped_capabilities(runner_permissions=list(registration.permissions or [])),
            approved_by=approved_by,
        )

    def _find_runner_registration(self, plugin_id: str, runner_id: str) -> PluginRunnerRegistration | None:
        plugin_id = str(plugin_id or "").strip()
        runner_id = str(runner_id or "").strip()
        for item in self.collect_sdk_registrations().get("runners", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("plugin_id") or "").strip() != plugin_id:
                continue
            try:
                registration = PluginRunnerRegistration.model_validate(item)
            except Exception:
                continue
            if registration.runner_id == runner_id or f"plugin.{registration.plugin_id}.{registration.runner_id}" == runner_id:
                return registration
        return None

    def _runner_approval_snapshot(
        self,
        registration: PluginRunnerRegistration | None,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if registration is None:
            return {
                "schema": "plugin-runner-approval-snapshot-v1",
                "approved": False,
                "approval_reason": "runner_registration_unavailable",
            }
        manifest = self._manifests.get(registration.plugin_id)
        package_hash = self._package_hashes.get(registration.plugin_id, "")
        return build_runner_approval_snapshot(
            plugin_id=registration.plugin_id,
            runner_id=registration.runner_id,
            request_schema_id=registration.request_schema_id,
            entrypoint=registration.entrypoint,
            runner_permissions=list(registration.permissions or []),
            context_metadata=context_metadata or {},
            approval_store=self._approval,
            manifest=manifest,
            package_hash=package_hash,
            developer_mode=self._developer_mode,
        )

    def run_sdk_runner(
        self,
        registration: PluginRunnerRegistration,
        request: BaseRequest,
        context: RunContext,
    ) -> RunResult:
        """Execute a loaded plugin runner through the narrow SDK harness."""
        context = self._with_sdk_host_policy(context)
        return execute_plugin_sdk_runner(
            registration=registration,
            request=request,
            context=context,
            find_plugin_dir=self._find_plugin_dir,
            collect_sdk_registrations=self.collect_sdk_registrations,
            approval_snapshot_builder=self._runner_approval_snapshot,
            read_raw_manifest_payload=self._read_raw_manifest_payload,
            audit_append=self._audit.append,
            manifest_loader=load_manifest,
            function_loader=load_plugin_functions,
        )

    def _sdk_host_policy_metadata(self) -> dict[str, Any]:
        return build_plugin_sdk_host_policy(developer_mode=self._developer_mode).context_metadata()

    def _with_sdk_host_policy(self, context: RunContext) -> RunContext:
        metadata = dict(self._sdk_host_policy_metadata())
        metadata.update(dict(context.metadata or {}))
        return RunContext(
            project_root=context.project_root,
            backend_root=context.backend_root,
            work_dir=context.work_dir,
            safe_roots=context.safe_roots,
            runtime_id=context.runtime_id,
            env=context.env,
            metadata=metadata,
        )

    def _find_plugin_dir(self, plugin_id: str) -> Path | None:
        return find_plugin_dir(self._plugin_root, plugin_id)

    def _approved_runner_permissions(self, registration: PluginRunnerRegistration) -> list[str]:
        manifest = self._manifests.get(registration.plugin_id)
        pkg_hash = self._package_hashes.get(registration.plugin_id)
        if manifest is None or not pkg_hash:
            return []
        signer = manifest.signature.signer if manifest.signature else ""
        identity = build_plugin_identity(registration.plugin_id, manifest.version, pkg_hash, signer)
        approval = self._approval.check(
            approval_key=identity,
            required_capabilities=runner_required_capabilities(
                manifest=manifest,
                runner_permissions=list(registration.permissions or []),
            ),
        )
        if not bool(approval.get("approved")):
            return []
        return list(registration.permissions or [])

    def get_plugin_settings(self, plugin_id: str) -> dict[str, Any]:
        """Return a plugin's settings schema, defaults, and saved values."""
        plugin_id = str(plugin_id or "").strip()
        return build_plugin_settings_payload(
            plugin_id=plugin_id,
            manifest_payload=self._read_raw_manifest_payload(plugin_id),
            all_settings=self._read_settings_all(),
            plugin_root=self._plugin_root,
            plugin_dir=self._find_plugin_dir(plugin_id),
        )

    def set_plugin_settings(self, plugin_id: str, settings: dict[str, Any], updated_by: str = "local-user") -> dict[str, Any]:
        """Persist user-facing settings for a plugin."""
        plugin_id = str(plugin_id or "").strip()
        current = self.get_plugin_settings(plugin_id)
        schema = current.get("schema", {}) if isinstance(current, dict) else {}
        sanitized = sanitize_plugin_settings(schema, settings)
        self._settings_store.set_plugin_settings(plugin_id, sanitized)
        self._audit.append(
            event_type="plugin_settings_changed",
            plugin_id=plugin_id,
            payload={"updated_by": updated_by, "keys": sorted(sanitized.keys())},
        )
        return {"success": True, "settings": self.get_plugin_settings(plugin_id)}

    def get_bus(self) -> EventBus:
        """Return the underlying EventBus."""
        return self._bus

    def get_orchestrator(self) -> PluginOrchestrator:
        """Return the underlying PluginOrchestrator."""
        return self._orchestrator

    # ── Snapshot persistence ───────────────────────────────────────────

    def _persist_training_snapshot(self) -> None:
        """Write training_hooks_snapshot.json for fast-path checks."""
        write_training_hooks_snapshot(self._snapshot_path, self._bus, self._orchestrator)


# ── Singleton accessor ─────────────────────────────────────────────────

_runtime_instance: PluginRuntime | None = None
_runtime_lock = threading.Lock()


def get_plugin_runtime() -> PluginRuntime:
    """Return the global PluginRuntime singleton (thread-safe lazy init)."""
    global _runtime_instance
    if _runtime_instance is not None:
        return _runtime_instance
    with _runtime_lock:
        if _runtime_instance is None:
            _runtime_instance = PluginRuntime()
        return _runtime_instance

