"""Friendly CLI for IDI workflows (build / verify agent packs, ZK bundles)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from idi.pipeline.agent_pack import build_agent_pack
from idi.zk.proof_manager import ProofBundle, verify_proof
from idi.zk.verification import VerificationErrorCode

SUCCESS = "✅"
STEP = "🚀"
WARN = "⚠️"
ERROR = "❌"


def _poke_yoke_path(path: Path, must_exist: bool = True) -> None:
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{ERROR} Path not found: {path}")


def _print_header(title: str) -> None:
    print(f"{STEP} {title}")


def _cmd_build(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    out_dir = Path(args.out)
    try:
        _poke_yoke_path(config_path, must_exist=True)
        _print_header("Building agent pack")
        build_agent_pack(
            config_path=config_path,
            out_dir=out_dir,
            spec_kind=args.spec_kind,
            proof_enabled=not args.no_proof,
            prover_cmd=args.prover_cmd,
            tau_bin=None,
            tx_hash=args.tx_hash,
        )
        print(f"{SUCCESS} Agent pack created at {out_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Build failed: {exc}")
        return 1


def _cmd_patch_validate(args: argparse.Namespace) -> int:
    try:
        from idi.synth import load_agent_patch

        patch_path = Path(args.patch)
        _poke_yoke_path(patch_path, must_exist=True)
        patch = load_agent_patch(patch_path)
        _print_header(f"Valid AgentPatch at {patch_path}")
        summary = {
            "id": patch.meta.id,
            "name": patch.meta.name,
            "agent_type": patch.agent_type,
            "tags": list(patch.meta.tags),
        }
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Patch validation failed: {exc}")
        return 1


def _cmd_patch_diff(args: argparse.Namespace) -> int:
    try:
        from idi.synth import diff_agent_patches, load_agent_patch

        old_path = Path(args.old)
        new_path = Path(args.new)
        _poke_yoke_path(old_path, must_exist=True)
        _poke_yoke_path(new_path, must_exist=True)

        old_patch = load_agent_patch(old_path)
        new_patch = load_agent_patch(new_path)

        _print_header(f"Diffing patches {old_path} vs {new_path}")
        diff = diff_agent_patches(old_patch, new_patch)
        print(json.dumps(diff, sort_keys=True, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Patch diff failed: {exc}")
        return 1


def _cmd_patch_create(args: argparse.Namespace) -> int:
    try:
        from idi.synth import AgentPatch, AgentPatchMeta, save_agent_patch

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tags = tuple(args.tag or ())
        meta = AgentPatchMeta(
            id=args.id,
            name=args.name,
            description=args.description,
            version=args.version,
            tags=tags,
        )
        patch = AgentPatch(
            meta=meta,
            agent_type=args.agent_type,
        )

        save_agent_patch(patch, out_path)
        _print_header(f"Created AgentPatch at {out_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Patch create failed: {exc}")
        return 1


def _cmd_patch_apply(args: argparse.Namespace) -> int:
    try:
        from idi.synth import agent_patch_to_dict, load_agent_patch, validate_agent_patch

        patch_path = Path(args.patch)
        _poke_yoke_path(patch_path, must_exist=True)
        patch = load_agent_patch(patch_path)
        validate_agent_patch(patch)
        _print_header(f"AgentPatch summary for {patch_path}")
        data = agent_patch_to_dict(patch)
        print(json.dumps(data, sort_keys=True, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Patch apply failed: {exc}")
        return 1


def _load_bundle(pack_dir: Path) -> ProofBundle:
    proof_dir = pack_dir / "proof"
    manifest_path = pack_dir / "artifact_manifest.json"
    return ProofBundle(
        manifest_path=manifest_path,
        proof_path=proof_dir / "proof.bin",
        receipt_path=proof_dir / "receipt.json",
        stream_dir=pack_dir / "streams",
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    pack_dir = Path(args.agentpack)
    try:
        _poke_yoke_path(pack_dir, must_exist=True)
        bundle = _load_bundle(pack_dir)
        for required in [bundle.manifest_path, bundle.proof_path, bundle.receipt_path, bundle.stream_dir]:
            _poke_yoke_path(required, must_exist=True)

        _print_header(f"Verifying agent pack at {pack_dir}")
        ok = verify_proof(bundle, use_risc0=args.risc0)
        if not ok:
            print(f"{ERROR} Verification failed")
            return 2
        print(f"{SUCCESS} Verification succeeded")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Verify failed: {exc}")
        return 1


def _cmd_bundle_pack(args: argparse.Namespace) -> int:
    """Pack a local proof into a network-portable wire bundle."""
    from idi.zk.wire import ZkProofBundleLocal
    
    proof_dir = Path(args.proof_dir)
    out_path = Path(args.out)
    
    try:
        _poke_yoke_path(proof_dir, must_exist=True)
        
        # Locate required files
        proof_path = proof_dir / "proof.bin"
        attestation_path = proof_dir / "receipt.json"
        manifest_path = proof_dir.parent / "artifact_manifest.json"
        if not manifest_path.exists():
            manifest_path = proof_dir / "manifest.json"
        stream_dir = proof_dir.parent / "streams"
        if not stream_dir.exists():
            stream_dir = proof_dir / "streams"
        
        for p in [proof_path, attestation_path, manifest_path]:
            _poke_yoke_path(p, must_exist=True)
        
        _print_header(f"Packing wire bundle from {proof_dir}")
        
        local = ZkProofBundleLocal(
            proof_path=proof_path,
            attestation_path=attestation_path,
            manifest_path=manifest_path,
            stream_dir=stream_dir if stream_dir.exists() else proof_dir,
            tx_hash=args.tx_hash,
        )
        
        wire = local.to_wire(include_streams=not args.no_streams)
        
        # Write wire bundle
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wire.serialize())
        
        print(f"{SUCCESS} Wire bundle created: {out_path}")
        print(f"    Schema: {wire.schema_version}")
        print(f"    Proof system: {wire.proof_system}")
        print(f"    Streams included: {wire.streams_pack_b64 is not None}")
        return 0
        
    except Exception as exc:
        print(f"{ERROR} Pack failed: {exc}")
        return 1


def _cmd_bundle_verify(args: argparse.Namespace) -> int:
    """Verify a wire bundle."""
    from idi.zk.wire import ZkProofBundleWireV1
    from idi.zk.bundle_verify import verify_proof_bundle_wire
    
    bundle_path = Path(args.bundle)
    
    try:
        _poke_yoke_path(bundle_path, must_exist=True)
        
        _print_header(f"Verifying wire bundle: {bundle_path}")
        
        # Load and deserialize
        data = bundle_path.read_bytes()
        wire = ZkProofBundleWireV1.deserialize(data)
        
        print(f"    Schema: {wire.schema_version}")
        print(f"    Proof system: {wire.proof_system}")
        
        # Verify
        report = verify_proof_bundle_wire(
            wire,
            expected_method_id=args.method_id,
            require_zk=args.require_zk,
        )

        # Experimental KRR (ZK/Tau invariants) annotation
        try:
            from idi.devkit.experimental.zk_krr_integration import (
                BundleStatusContext,
                evaluate_bundle_with_krr,
            )

            code = report.error_code
            success = bool(report.success)

            def _ok_unless(code_match: VerificationErrorCode) -> bool:
                return success or code != code_match

            commitment_ok = _ok_unless(VerificationErrorCode.COMMITMENT_MISMATCH)
            method_id_ok = _ok_unless(VerificationErrorCode.METHOD_ID_MISMATCH)
            journal_ok = _ok_unless(VerificationErrorCode.JOURNAL_DIGEST_MISMATCH)
            tx_hash_ok = _ok_unless(VerificationErrorCode.TX_HASH_MISMATCH)

            path_ok = True
            if not success and code in {
                VerificationErrorCode.STREAMS_DIGEST_MISMATCH,
                VerificationErrorCode.STREAMS_MISSING,
                VerificationErrorCode.RECEIPT_PARSE_ERROR,
            }:
                path_ok = False

            ctx = BundleStatusContext(
                bundle_id=str(bundle_path),
                commitment_ok=commitment_ok,
                method_id_ok=method_id_ok,
                journal_ok=journal_ok,
                tx_hash_ok=tx_hash_ok,
                path_ok=path_ok,
            )

            allowed, reasons = evaluate_bundle_with_krr(ctx)
            print(f"{STEP} ZK/Tau invariants (KRR) evaluation:")
            if allowed:
                print(f"    {SUCCESS} All encoded invariants satisfied under current status flags")
            else:
                print(f"    {WARN} Invariant violations detected by KRR:")
                for reason in reasons:
                    print(f"      - {reason}")
        except Exception:
            print(f"{WARN} Skipping experimental KRR annotation (STRIKE/IKL layer unavailable)")

        if report.success:
            print(f"{SUCCESS} Verification succeeded")
            if report.details.get("digest"):
                print(f"    Commitment: {report.details['digest'][:16]}...")
            return 0
        else:
            print(f"{ERROR} Verification failed: {report.error_code.value}")
            print(f"    {report.message}")
            return 2
            
    except ValueError as exc:
        print(f"{ERROR} Invalid bundle: {exc}")
        return 1
    except Exception as exc:
        print(f"{ERROR} Verify failed: {exc}")
        return 1


def _cmd_bundle_info(args: argparse.Namespace) -> int:
    """Display information about a wire bundle."""
    from idi.zk.wire import ZkProofBundleWireV1
    
    bundle_path = Path(args.bundle)
    
    try:
        _poke_yoke_path(bundle_path, must_exist=True)
        
        data = bundle_path.read_bytes()
        wire = ZkProofBundleWireV1.deserialize(data)
        
        print(f"Wire Bundle: {bundle_path}")
        print(f"  Schema version: {wire.schema_version}")
        print(f"  Proof system: {wire.proof_system}")
        print(f"  Has streams: {wire.streams_pack_b64 is not None}")
        if wire.streams_sha256:
            print(f"  Streams SHA256: {wire.streams_sha256[:16]}...")
        if wire.tx_hash:
            print(f"  TX hash: {wire.tx_hash}")
        
        # Compute commitment
        commitment = wire.compute_commitment()
        print(f"  Commitment: {commitment[:32]}...")
        
        # Parse attestation
        try:
            attestation = wire.get_attestation()
            print(f"  Attestation keys: {list(attestation.keys())}")
        except Exception:
            print(f"  {WARN} Could not parse attestation")
        
        return 0
        
    except Exception as exc:
        print(f"{ERROR} Info failed: {exc}")
        return 1


def _cmd_dev_sape_qpatch(args: argparse.Namespace) -> int:
    """Run experimental SAPE evolution over a minimal QAgentPatch space.

    This is a developer-only command intended for experimentation with the
    Spec-Aware Patch Evolution (SAPE) algorithm. It does not integrate with
    production workflows.
    """
    try:
        from idi.devkit.experimental.sape_q_patch import (
            QAgentPatch,
            QPatchMeta,
            evolve_q_patches,
            evaluate_patch_real,
            evaluate_patch_stub,
        )

        base = QAgentPatch(
            identifier="base",
            num_price_bins=args.price_bins,
            num_inventory_bins=args.inventory_bins,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            meta=QPatchMeta(
                name="dev-sape-qpatch",
                description="Experimental SAPE Q-agent patch",
                version="0.1.0",
                tags=("experimental", "qtable"),
            ),
        )

        _print_header("Running experimental SAPE Q-patch evolution")

        if getattr(args, "eval_mode", "real") == "synthetic":
            evaluator = evaluate_patch_stub
        else:
            evaluator = evaluate_patch_real

        population = evolve_q_patches(
            base_patch=base,
            population_size=args.population_size,
            iterations=args.iterations,
            evaluator=evaluator,
        )

        print(f"{SUCCESS} Found {len(population)} Pareto patches")
        for entry in population:
            patch = entry.patch
            metrics_str = ", ".join(f"{v:.4f}" for v in entry.values)
            print(
                "  id="
                f"{patch.identifier} "
                f"bins={patch.num_price_bins}x{patch.num_inventory_bins} "
                f"lr={patch.learning_rate:.4f} "
                f"eps_start={patch.epsilon_start:.4f} "
                f"metrics=[{metrics_str}]",
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Experimental SAPE run failed: {exc}")
        return 1


def _cmd_dev_qagent_synth(args: argparse.Namespace) -> int:
    try:
        from idi.devkit.experimental.qagent_synth import (
            QAgentSynthConfig,
            QAgentSynthesizer,
            load_qagent_patch_preset,
        )
        from idi.devkit.experimental.sape_q_patch import (
            QAgentPatch,
            QPatchMeta,
            evaluate_patch_real,
            evaluate_patch_stub,
        )

        if args.preset:
            preset_path = Path(args.preset)
            _poke_yoke_path(preset_path, must_exist=True)
            base = load_qagent_patch_preset(preset_path)
        else:
            base = QAgentPatch(
                identifier="base",
                num_price_bins=args.price_bins,
                num_inventory_bins=args.inventory_bins,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                meta=QPatchMeta(
                    name="dev-qagent-synth",
                    description="Experimental QAgent modular synth patch",
                    version="0.1.0",
                    tags=("experimental", "qtable"),
                ),
            )

        if getattr(args, "eval_mode", "synthetic") == "real":
            evaluator = evaluate_patch_real
        else:
            evaluator = evaluate_patch_stub

        profiles = {"conservative"}
        cfg = QAgentSynthConfig(
            beam_width=args.beam_width,
            max_depth=args.max_depth,
        )
        synth = QAgentSynthesizer(
            base,
            profiles=profiles,
            evaluator=evaluator,
        )

        _print_header("Running experimental QAgent modular synth search")
        results = synth.synthesize(config=cfg)

        print(f"{SUCCESS} Found {len(results)} candidate patches")
        for patch, score in results:
            score_str = ", ".join(f"{v:.4f}" for v in score)
            print(
                "  id="
                f"{patch.identifier} "
                f"bins={patch.num_price_bins}x{patch.num_inventory_bins} "
                f"lr={patch.learning_rate:.4f} "
                f"eps_start={patch.epsilon_start:.4f} "
                f"metrics=[{score_str}]",
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} QAgent modular synth run failed: {exc}")
        return 1


def _cmd_dev_comm_krr(args: argparse.Namespace) -> int:
    try:
        from idi_iann.config import TrainingConfig
        from idi.devkit.experimental.comm_krr_trainer import KRRWrappedQTrainer

        cfg: TrainingConfig
        if args.config:
            cfg_path = Path(args.config)
            _poke_yoke_path(cfg_path, must_exist=True)
            cfg = TrainingConfig.from_json(cfg_path)
        else:
            cfg = TrainingConfig(episodes=args.episodes)

        seed = None if args.seed == -1 else args.seed

        trainer = KRRWrappedQTrainer(
            config=cfg,
            use_crypto_env=args.use_crypto_env,
            seed=seed,
            subject_id=args.subject_id,
            sensitivity=args.sensitivity,
        )
        trainer.run()
        stats = trainer.stats()
        _print_header("Communication KRR statistics")
        print(json.dumps(stats, sort_keys=True, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Experimental comm KRR run failed: {exc}")
        return 1


def _cmd_dev_auto_qagent(args: argparse.Namespace) -> int:
    """Run minimal Auto-QAgent synthesis using QAgentSynthesizer.

    This command is experimental and currently focuses on the
    design/synthesis phase. It uses a synthetic evaluator and does not
    orchestrate full training or proof generation yet.
    """
    try:
        from idi.devkit.experimental.auto_qagent import (
            load_goal_spec,
            run_auto_qagent_synth,
        )

        goal_path = Path(args.goal)
        _poke_yoke_path(goal_path, must_exist=True)

        goal = load_goal_spec(goal_path)
        _print_header("Running Auto-QAgent synthesis")
        results = run_auto_qagent_synth(goal)

        print(f"{SUCCESS} Found {len(results)} candidate patches")
        for patch, score in results:
            score_str = ", ".join(f"{v:.4f}" for v in score)
            print(
                "  id="
                f"{patch.identifier} "
                f"bins={patch.num_price_bins}x{patch.num_inventory_bins} "
                f"lr={patch.learning_rate:.4f} "
                f"eps_start={patch.epsilon_start:.4f} "
                f"metrics=[{score_str}]",
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{ERROR} Auto-QAgent synthesis failed: {exc}")
        return 1


# -----------------------------------------------------------------------------
# UX / Parameterization Commands
# -----------------------------------------------------------------------------


def _cmd_presets(args: argparse.Namespace) -> int:
    """List available goal spec presets."""
    try:
        from idi.gui.backend.services.presets import PresetService
        
        service = PresetService()
        presets = service.list_all()
        
        # Filter by tag if specified
        if args.tag:
            presets = [p for p in presets if args.tag in p.tags]
        
        if args.json:
            import json
            data = [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "icon": p.icon,
                    "tags": list(p.tags),
                    "difficulty": p.difficulty,
                }
                for p in presets
            ]
            print(json.dumps(data, indent=2))
        else:
            _print_header("Available Presets")
            for p in presets:
                icon = {"shield": "🛡️", "flask": "🔬", "zap": "⚡", "check-circle": "✅"}.get(p.icon, "📦")
                print(f"  {icon} {p.id}")
                print(f"     Name: {p.name}")
                print(f"     {p.description}")
                print(f"     Tags: {', '.join(p.tags)}")
                print()
        return 0
    except Exception as exc:
        print(f"{ERROR} Failed to list presets: {exc}")
        return 1


def _cmd_preset_show(args: argparse.Namespace) -> int:
    """Show details of a specific preset."""
    try:
        from idi.gui.backend.services.presets import PresetService
        import json
        
        service = PresetService()
        preset = service.get(args.preset_id)
        
        if not preset:
            print(f"{ERROR} Preset not found: {args.preset_id}")
            return 1
        
        goal_spec = service.load_goal_spec(args.preset_id)
        
        if args.json:
            print(json.dumps(goal_spec, indent=2))
        else:
            _print_header(f"Preset: {preset.name}")
            print(f"  ID: {preset.id}")
            print(f"  Description: {preset.description}")
            print(f"  Difficulty: {preset.difficulty}")
            print(f"  Tags: {', '.join(preset.tags)}")
            print()
            print("Goal Spec:")
            print(json.dumps(goal_spec, indent=2))
        return 0
    except Exception as exc:
        print(f"{ERROR} Failed to show preset: {exc}")
        return 1


def _cmd_macros(args: argparse.Namespace) -> int:
    """List available macro controls."""
    try:
        from idi.gui.backend.services.macros import MacroService
        
        service = MacroService()
        macros = service.list_all()
        
        if args.json:
            import json
            data = [
                {
                    "id": m.id,
                    "label": m.label,
                    "description": m.description,
                    "default": m.default,
                    "effects": m.effects,
                }
                for m in macros
            ]
            print(json.dumps(data, indent=2))
        else:
            _print_header("Available Macro Controls")
            for m in macros:
                print(f"  🎛️  {m.id}")
                print(f"     Label: {m.label}")
                print(f"     Description: {m.description}")
                print(f"     Default: {m.default:.1%}")
                print(f"     Affects: {', '.join(m.effects)}")
                print()
        return 0
    except Exception as exc:
        print(f"{ERROR} Failed to list macros: {exc}")
        return 1


def _cmd_macro_apply(args: argparse.Namespace) -> int:
    """Apply macro values to a goal spec."""
    try:
        from idi.gui.backend.services.macros import MacroService
        import json
        
        goal_path = Path(args.goal)
        _poke_yoke_path(goal_path, must_exist=True)
        
        base_spec = json.loads(goal_path.read_text())
        service = MacroService()
        
        # Collect macro values from args
        macro_values = {}
        if args.risk_appetite is not None:
            macro_values["risk_appetite"] = args.risk_appetite
        if args.exploration is not None:
            macro_values["exploration_intensity"] = args.exploration
        if args.training_time is not None:
            macro_values["training_time"] = args.training_time
        if args.conservatism is not None:
            macro_values["conservatism"] = args.conservatism
        if args.stability_reward is not None:
            macro_values["stability_reward"] = args.stability_reward
        
        if not macro_values:
            print(f"{WARN} No macro values specified. Use --risk-appetite, --exploration, etc.")
            return 1
        
        # Apply macros
        result = service.apply_all(macro_values, base_spec)
        output = json.dumps(result, indent=2)
        
        if args.out:
            out_path = Path(args.out)
            out_path.write_text(output)
            print(f"{SUCCESS} Wrote modified goal spec to {out_path}")
        else:
            print(output)
        
        return 0
    except Exception as exc:
        print(f"{ERROR} Failed to apply macros: {exc}")
        return 1


def _cmd_invariants(args: argparse.Namespace) -> int:
    """Check invariants for a goal spec."""
    try:
        from idi.gui.backend.services.invariants import InvariantService
        import json
        
        goal_path = Path(args.goal)
        _poke_yoke_path(goal_path, must_exist=True)
        
        goal_spec = json.loads(goal_path.read_text())
        service = InvariantService()
        results = service.check_all(goal_spec)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print_header("Invariant Check Results")
            all_ok = True
            for r in results:
                status = "✅" if r["ok"] else "❌"
                all_ok = all_ok and r["ok"]
                print(f"  {status} {r['id']}: {r['label']}")
                print(f"     {r['message']}")
            print()
            if all_ok:
                print(f"{SUCCESS} All invariants satisfied!")
            else:
                print(f"{WARN} Some invariants violated. Review configuration.")
        return 0 if all_ok else 1
    except Exception as exc:
        print(f"{ERROR} Failed to check invariants: {exc}")
        return 1


def _cmd_gui(args: argparse.Namespace) -> int:
    """Start the IDI Synth Studio GUI server."""
    try:
        import subprocess
        import webbrowser
        
        gui_backend_path = Path(__file__).parent / "gui" / "backend" / "main.py"
        
        if not gui_backend_path.exists():
            print(f"{ERROR} GUI backend not found at {gui_backend_path}")
            return 1
        
        _print_header("Starting IDI Synth Studio")
        print(f"  Backend: http://127.0.0.1:{args.port}")
        print(f"  Frontend: Run 'npm run dev' in idi/gui/frontend/")
        print()
        print("Press Ctrl+C to stop the server.")
        
        if not args.no_browser:
            # Try to open browser (will fail gracefully if frontend not running)
            try:
                webbrowser.open(f"http://localhost:5173")
            except Exception:
                pass
        
        # Start the backend server
        subprocess.run(
            ["python3", str(gui_backend_path)],
            cwd=gui_backend_path.parent,
            check=True,
        )
        return 0
    except KeyboardInterrupt:
        print(f"\n{SUCCESS} GUI server stopped.")
        return 0
    except Exception as exc:
        print(f"{ERROR} Failed to start GUI: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IDI agent pack CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Agent pack commands
    p_build = sub.add_parser("build", help="Train, export, prove, and bundle an agent pack")
    p_build.add_argument("--config", required=True, help="Path to training config JSON")
    p_build.add_argument("--out", required=True, help="Output directory for agent pack")
    p_build.add_argument("--spec-kind", default="v38", help="Spec type to generate")
    p_build.add_argument("--prover-cmd", help="Optional prover command for proofs")
    p_build.add_argument("--no-proof", action="store_true", help="Skip proof generation")
    p_build.add_argument("--tx-hash", help="Optional transaction hash binding")
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="Verify an existing agent pack (proof + receipt)")
    p_verify.add_argument("--agentpack", required=True, help="Path to built agent pack directory")
    p_verify.add_argument("--risc0", action="store_true", help="Use Risc0 verification path")
    p_verify.set_defaults(func=_cmd_verify)

    p_patch_validate = sub.add_parser("patch-validate", help="Validate an AgentPatch JSON file")
    p_patch_validate.add_argument("--patch", required=True, help="Path to AgentPatch JSON file")
    p_patch_validate.set_defaults(func=_cmd_patch_validate)

    p_patch_diff = sub.add_parser("patch-diff", help="Diff two AgentPatch JSON files")
    p_patch_diff.add_argument("--old", required=True, help="Path to original AgentPatch JSON file")
    p_patch_diff.add_argument("--new", required=True, help="Path to updated AgentPatch JSON file")
    p_patch_diff.set_defaults(func=_cmd_patch_diff)

    p_patch_create = sub.add_parser("patch-create", help="Create a skeleton AgentPatch JSON file")
    p_patch_create.add_argument("--out", required=True, help="Output path for AgentPatch JSON file")
    p_patch_create.add_argument("--id", default="patch", help="Patch identifier")
    p_patch_create.add_argument("--name", default="unnamed-patch", help="Human-readable name")
    p_patch_create.add_argument("--description", default="AgentPatch template", help="Short description")
    p_patch_create.add_argument("--version", default="0.0.1", help="Semantic version")
    p_patch_create.add_argument("--agent-type", default="generic", help="Agent type label")
    p_patch_create.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Tag to attach to the patch (may be repeated)",
    )
    p_patch_create.set_defaults(func=_cmd_patch_create)

    p_patch_apply = sub.add_parser("patch-apply", help="Validate and describe an AgentPatch JSON file")
    p_patch_apply.add_argument("--patch", required=True, help="Path to AgentPatch JSON file")
    p_patch_apply.set_defaults(func=_cmd_patch_apply)

    # ZK bundle commands
    p_bundle = sub.add_parser("bundle", help="ZK wire bundle operations")
    bundle_sub = p_bundle.add_subparsers(dest="bundle_command", required=True)
    
    p_pack = bundle_sub.add_parser("pack", help="Pack proof directory into wire bundle")
    p_pack.add_argument("--proof-dir", required=True, help="Directory containing proof.bin and receipt.json")
    p_pack.add_argument("--out", required=True, help="Output wire bundle JSON file")
    p_pack.add_argument("--tx-hash", help="Optional transaction hash binding")
    p_pack.add_argument("--no-streams", action="store_true", help="Exclude streams from bundle")
    p_pack.set_defaults(func=_cmd_bundle_pack)
    
    p_bverify = bundle_sub.add_parser("verify", help="Verify a wire bundle")
    p_bverify.add_argument("--bundle", required=True, help="Path to wire bundle JSON")
    p_bverify.add_argument("--method-id", help="Expected method ID (hex) for Risc0")
    p_bverify.add_argument("--require-zk", action="store_true", help="Require ZK verification")
    p_bverify.set_defaults(func=_cmd_bundle_verify)
    
    p_info = bundle_sub.add_parser("info", help="Display wire bundle information")
    p_info.add_argument("--bundle", required=True, help="Path to wire bundle JSON")
    p_info.set_defaults(func=_cmd_bundle_info)

    # Experimental / dev commands
    p_dev_sape = sub.add_parser(
        "dev-sape-qpatch",
        help="Run experimental SAPE evolution over a minimal QAgentPatch",
    )
    p_dev_sape.add_argument("--price-bins", type=int, default=10, help="Number of price bins")
    p_dev_sape.add_argument(
        "--inventory-bins",
        type=int,
        default=10,
        help="Number of inventory bins",
    )
    p_dev_sape.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Initial learning rate",
    )
    p_dev_sape.add_argument(
        "--discount-factor",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    p_dev_sape.add_argument(
        "--epsilon-start",
        type=float,
        default=0.5,
        help="Starting exploration rate",
    )
    p_dev_sape.add_argument(
        "--epsilon-end",
        type=float,
        default=0.1,
        help="Final exploration rate",
    )
    p_dev_sape.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=1000,
        help="Steps over which epsilon decays",
    )
    p_dev_sape.add_argument(
        "--population-size",
        type=int,
        default=4,
        help="Population (beam) size for SAPE",
    )
    p_dev_sape.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of SAPE iterations",
    )
    p_dev_sape.add_argument(
        "--eval-mode",
        choices=["real", "synthetic"],
        default="real",
        help="Evaluation mode: real QTrainer-based or synthetic metric",
    )
    p_dev_sape.set_defaults(func=_cmd_dev_sape_qpatch)

    p_dev_qsynth = sub.add_parser(
        "dev-qagent-synth",
        help="Run experimental QAgent modular synth search",
    )
    p_dev_qsynth.add_argument("--price-bins", type=int, default=10, help="Number of price bins")
    p_dev_qsynth.add_argument(
        "--inventory-bins",
        type=int,
        default=10,
        help="Number of inventory bins",
    )
    p_dev_qsynth.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Initial learning rate",
    )
    p_dev_qsynth.add_argument(
        "--discount-factor",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    p_dev_qsynth.add_argument(
        "--epsilon-start",
        type=float,
        default=0.5,
        help="Starting exploration rate",
    )
    p_dev_qsynth.add_argument(
        "--epsilon-end",
        type=float,
        default=0.1,
        help="Final exploration rate",
    )
    p_dev_qsynth.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=1000,
        help="Steps over which epsilon decays",
    )
    p_dev_qsynth.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for modular synth search",
    )
    p_dev_qsynth.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum search depth for modular synth",
    )
    p_dev_qsynth.add_argument(
        "--preset",
        help="Optional QAgent patch preset JSON",
    )
    p_dev_qsynth.add_argument(
        "--eval-mode",
        choices=["real", "synthetic"],
        default="synthetic",
        help="Evaluation mode: real QTrainer-based or synthetic metric",
    )
    p_dev_qsynth.set_defaults(func=_cmd_dev_qagent_synth)

    p_dev_comm = sub.add_parser(
        "dev-comm-krr",
        help="Run QTrainer with communication KRR gating (experimental)",
    )
    p_dev_comm.add_argument(
        "--config",
        help="Optional TrainingConfig JSON path",
    )
    p_dev_comm.add_argument(
        "--episodes",
        type=int,
        default=32,
        help="Episodes to train if no config is provided",
    )
    p_dev_comm.add_argument(
        "--use-crypto-env",
        action="store_true",
        help="Use the crypto market simulator",
    )
    p_dev_comm.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (-1 for nondeterministic)",
    )
    p_dev_comm.add_argument(
        "--subject-id",
        type=str,
        default="agent",
        help="Subject identifier for KRR facts",
    )
    p_dev_comm.add_argument(
        "--sensitivity",
        choices=["low", "high"],
        default="high",
        help="User sensitivity level for KRR facts",
    )
    p_dev_comm.set_defaults(func=_cmd_dev_comm_krr)

    p_dev_auto = sub.add_parser(
        "dev-auto-qagent",
        help="Run minimal Auto-QAgent synthesis (experimental)",
    )
    p_dev_auto.add_argument(
        "--goal",
        required=True,
        help="Path to Auto-QAgent goal spec JSON",
    )
    p_dev_auto.add_argument(
        "--out-patches",
        help="Optional directory to export AgentPatch JSON files",
    )
    p_dev_auto.set_defaults(func=_cmd_dev_auto_qagent)

    # -------------------------------------------------------------------------
    # UX / Parameterization Commands
    # -------------------------------------------------------------------------

    p_presets = sub.add_parser(
        "presets",
        help="List available goal spec presets",
    )
    p_presets.add_argument(
        "--tag",
        help="Filter presets by tag (e.g., 'beginner', 'low-risk')",
    )
    p_presets.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    p_presets.set_defaults(func=_cmd_presets)

    p_preset_show = sub.add_parser(
        "preset-show",
        help="Show details of a specific preset",
    )
    p_preset_show.add_argument(
        "preset_id",
        help="Preset ID to show (e.g., 'conservative_qagent')",
    )
    p_preset_show.add_argument(
        "--json",
        action="store_true",
        help="Output goal spec as JSON",
    )
    p_preset_show.set_defaults(func=_cmd_preset_show)

    p_macros = sub.add_parser(
        "macros",
        help="List available macro controls",
    )
    p_macros.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    p_macros.set_defaults(func=_cmd_macros)

    p_macro_apply = sub.add_parser(
        "macro-apply",
        help="Apply macro values to a goal spec",
    )
    p_macro_apply.add_argument(
        "--goal",
        required=True,
        help="Path to base goal spec JSON",
    )
    p_macro_apply.add_argument(
        "--out",
        help="Output path for modified goal spec (defaults to stdout)",
    )
    p_macro_apply.add_argument(
        "--risk-appetite",
        type=float,
        help="Risk appetite (0.0-1.0)",
    )
    p_macro_apply.add_argument(
        "--exploration",
        type=float,
        help="Exploration intensity (0.0-1.0)",
    )
    p_macro_apply.add_argument(
        "--training-time",
        type=float,
        help="Training time emphasis (0.0-1.0)",
    )
    p_macro_apply.add_argument(
        "--conservatism",
        type=float,
        help="Conservatism level (0.0-1.0)",
    )
    p_macro_apply.add_argument(
        "--stability-reward",
        type=float,
        help="Stability vs reward balance (0.0-1.0)",
    )
    p_macro_apply.set_defaults(func=_cmd_macro_apply)

    p_invariants = sub.add_parser(
        "invariants",
        help="Check invariants for a goal spec",
    )
    p_invariants.add_argument(
        "--goal",
        required=True,
        help="Path to goal spec JSON",
    )
    p_invariants.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    p_invariants.set_defaults(func=_cmd_invariants)

    p_gui = sub.add_parser(
        "gui",
        help="Start the IDI Synth Studio GUI server",
    )
    p_gui.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the backend server",
    )
    p_gui.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    p_gui.set_defaults(func=_cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
