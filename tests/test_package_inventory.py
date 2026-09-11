from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools import verify_package_inventory as inventory
from tools.verify_package_inventory import (
    INTENTIONALLY_CHANGED_BINDINGS,
    LOCAL_IMPORT_SEAM_MARKER,
    MANIFEST,
    REVIEWED_V20_ADDED_DEFINITIONS,
    REVIEWED_V20_ADDED_STATEMENTS,
    azimuthal_rename_replacements,
    digest,
    main as verify_inventory,
    reviewed_local_import_seams,
    stable_ast_dump,
)


def inspect_seams(source: str):
    source = textwrap.dedent(source)
    return reviewed_local_import_seams("sample", source, ast.parse(source))


class PackageInventoryTests(unittest.TestCase):
    def test_lta_release_authenticates_runtime_version_bindings(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        manifest['v21_1_review']['statements'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21.1.0 review digest mismatch'):
            inventory.reviewed_v21_1_contract(
                manifest, manifest['v21_review'],
                *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
            )

    def test_lta_release_requires_latest_version_predecessor(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        review = manifest['v21_1_review']
        review['statements'][0]['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(
            review, sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_1_SHA256', authenticated):
            with self.assertRaisesRegex(
                RuntimeError, 'v21.1.0 supersession does not match its historical pin',
            ):
                inventory.reviewed_v21_1_contract(
                    manifest, manifest['v21_review'],
                    *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
                )

    def test_release_authenticates_upload_and_observability(self) -> None:
        for category, target in (
            ('definitions', 'RadialCudaProjector'),
            ('definitions', 'backproject_spherical_volume_to_volume'),
            ('definitions', '_execution_runtime_provenance'),
            ('preserved_radial_definition_updates', None),
            ('preserved_radial_module_updates', None),
        ):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            records = manifest['v21_0_6_review'][category]
            record = records[0] if target is None else next(item for item in records if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(category=category, target=target), self.assertRaisesRegex(
                    RuntimeError, 'v21.0.6 review digest mismatch'):
                inventory.reviewed_v21_0_6_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 6)))

    def test_release_requires_the_preserved_radial_class_predecessor(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        review = manifest['v21_0_6_review']
        next(item for item in review['definitions'] if item['name'] == 'RadialCudaProjector')['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_6_SHA256', authenticated), self.assertRaisesRegex(
                RuntimeError, 'v21.0.6 supersession does not match its historical pin'):
            inventory.reviewed_v21_0_6_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 6)))

    def test_release_requires_exact_preserved_module_and_upload_predecessors(self) -> None:
        for category, validate in (
            ('preserved_radial_module_updates', inventory.reviewed_radial_module_hashes),
            ('preserved_radial_definition_updates', inventory.reviewed_radial_definition_hashes),
        ):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            patches = [manifest[f'v21_0_{i}_review'] for i in range(1, 7)]
            patches[-1][category][0]['previous_sha256'] = '0' * 64
            with self.subTest(category=category), self.assertRaisesRegex(RuntimeError, 'preserved predecessor'):
                validate(manifest['v21_review'], patches)

    def test_release_does_not_allow_unknown_preserved_modules(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        patches = [manifest[f'v21_0_{i}_review'] for i in range(1, 7)]
        patches[-1]['preserved_radial_module_updates'][0]['module'] = 'unreviewed_module'
        with self.assertRaisesRegex(RuntimeError, 'unknown, duplicate or unexplained Radial module'):
            inventory.reviewed_radial_module_hashes(manifest['v21_review'], patches)

    def test_source_upload_method_has_an_independent_review_pin(self) -> None:
        original_digest = inventory.digest
        def altered_upload(node):
            return '0' * 64 if getattr(node, 'name', None) == '_upload_cropped_source' else original_digest(node)
        with mock.patch.object(inventory, 'digest', side_effect=altered_upload), self.assertRaisesRegex(
                RuntimeError, 'preserved Radial arithmetic: RadialCudaProjector._upload_cropped_source'):
            verify_inventory()

    def test_release_pins_native_ring_and_observability_contracts(self) -> None:
        for target in ('_ResidentTensorRTRingExecutor', 'GpuRenderedYoloSource',
                       '_claim_specialized_prediction_targets', 'RuntimeTelemetry', 'TtaScheduler'):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            review = manifest['v21_0_6_review']
            record = next(item for item in review['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.6 review digest mismatch'):
                inventory.reviewed_v21_0_6_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 6)))

    def test_release_authenticates_compact_projection_and_age_admission(self) -> None:
        for target in ('_MainProcessGpuStageCoordinator', '_project_spherical_encoded_block',
                       '_pull_spherical_f64'):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            review = manifest['v21_0_5_review']
            record = next(item for item in review['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.5 review digest mismatch'):
                inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_release_checks_current_compiled_kernel(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        review = manifest['v21_0_5_review']
        record = next(item for item in review['definitions'] if item['name'] == '_pull_spherical_f64')
        self.assertIsNone(record['previous_sha256'])
        original_digest = inventory.digest

        def altered_kernel(node):
            return '0' * 64 if getattr(node, 'name', None) == '_pull_spherical_f64' else original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=altered_kernel), self.assertRaisesRegex(
                RuntimeError, 'v21 patch reviewed definition changed or is missing: spherical_projection_cpu._pull_spherical_f64'):
            verify_inventory()

    def test_release_authenticates_fast_geometry_and_preserved_radial_update(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        review = manifest['v21_0_5_review']
        update = review['preserved_radial_definition_updates'][0]
        update['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21.0.5 review digest mismatch'):
            inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_preserved_radial_update_requires_the_exact_predecessor(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        patches = [manifest[f'v21_0_{i}_review'] for i in range(1, 6)]
        patches[-1]['preserved_radial_definition_updates'][0]['previous_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'preserved predecessor'):
            inventory.reviewed_radial_definition_hashes(manifest['v21_review'], patches)

    def test_release_authenticates_scheduling_and_cancelled_reader_lifetime(self) -> None:
        for target in ('TtaScheduler', '_MainProcessGpuStageCoordinator', '_ordered_spherical_blocks'):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            record = next(item for item in manifest['v21_0_5_review']['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.5 review digest mismatch'):
                inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_release_cannot_skip_the_latest_retirement_coordinator(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        fifth = manifest['v21_0_5_review']
        record = next(item for item in fifth['definitions'] if item['name'] == '_MainProcessGpuStageCoordinator')
        record['previous_sha256'] = '0' * 64
        reauthenticated = hashlib.sha256(json.dumps(fifth, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_5_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.5 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_fourth_patch_authenticates_geometry_projection_and_compaction(self) -> None:
        for target in ('qsc_inverse', '_project_spherical_block', '_resident_mask_kernels'):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            record = next(item for item in manifest['v21_0_4_review']['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.4 review digest mismatch'):
                inventory.reviewed_v21_0_4_contract(
                    manifest, manifest['v21_review'], manifest['v21_0_1_review'],
                    manifest['v21_0_2_review'], manifest['v21_0_3_review'],
                )

    def test_fourth_patch_cannot_skip_the_latest_diagnostic_revision(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        fourth = manifest['v21_0_4_review']
        record = next(item for item in fourth['definitions'] if item['name'] == '_announce_direct_compaction_layout')
        record['previous_sha256'] = '0' * 64
        reauthenticated = hashlib.sha256(json.dumps(
            fourth, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_4_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.4 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_4_contract(
                    manifest, manifest['v21_review'], manifest['v21_0_1_review'],
                    manifest['v21_0_2_review'], manifest['v21_0_3_review'],
                )

    def test_third_patch_cannot_skip_the_latest_coordinator_revision(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        prior = inventory.reviewed_v21_contract(manifest)
        first = inventory.reviewed_v21_patch_contract(manifest, prior)
        second = inventory.reviewed_v21_0_2_contract(manifest, prior, first)
        third = inventory.reviewed_v21_0_3_contract(manifest, prior, first, second)
        self.assertEqual(third['previous_review_sha256'], inventory.REVIEWED_V21_0_2_SHA256)
        earlier = next(item for item in first['definitions']
                       if item['name'] == '_MainProcessGpuStageCoordinator')
        current = next(item for item in third['definitions']
                       if item['name'] == '_MainProcessGpuStageCoordinator')
        current['previous_sha256'] = earlier['sha256']
        reauthenticated = hashlib.sha256(json.dumps(
            third, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_3_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.3 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_3_contract(manifest, prior, first, second)

    def test_third_patch_authenticates_preflight_and_layout_diagnostics(self) -> None:
        for target in ('validate_spherical_preflight_plane', '_announce_direct_compaction_layout'):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            record = next(item for item in manifest['v21_0_3_review']['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.3 review digest mismatch'):
                inventory.reviewed_v21_0_3_contract(
                    manifest, manifest['v21_review'], manifest['v21_0_1_review'], manifest['v21_0_2_review'],
                )

    def test_third_patch_checks_the_runtime_preflight_pixel_budget(self) -> None:
        original_digest = inventory.digest

        def changed_budget(node):
            targets = getattr(node, 'targets', ())
            if any(getattr(target, 'id', None) == '_PROBE_PIXELS' for target in targets):
                return '0' * 64
            return original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=changed_budget):
            with self.assertRaisesRegex(RuntimeError, 'reviewed statement changed or is missing: spherical_preflight.binding__PROBE_PIXELS'):
                verify_inventory()

    def test_second_patch_cannot_skip_the_intermediate_release(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        prior = inventory.reviewed_v21_contract(manifest)
        first = inventory.reviewed_v21_patch_contract(manifest, prior)
        second = inventory.reviewed_v21_0_2_contract(manifest, prior, first)
        self.assertEqual(second['previous_review_sha256'], inventory.REVIEWED_V21_0_1_SHA256)
        old_record = next(item for item in first['definitions']
                          if item['name'] == '_MainProcessGpuStageCoordinator')
        record = next(item for item in second['definitions']
                      if item['name'] == '_MainProcessGpuStageCoordinator')
        record['previous_sha256'] = old_record['previous_sha256']
        reauthenticated = hashlib.sha256(json.dumps(
            second, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_2_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.2 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_2_contract(manifest, prior, first)
        manifest['v21_0_1_review']['release'] = '21.0.2'
        with self.assertRaisesRegex(RuntimeError, 'v21.0.1 review digest mismatch'):
            inventory.reviewed_v21_patch_contract(manifest, prior)

    def test_second_patch_pins_current_compaction_and_bounds_definitions(self) -> None:
        original_digest = inventory.digest
        for target in ('_build_direct_device_compacted_payload', 'GpuFlattenedRetinaPayload', 'spherical_output_bounds'):
            def changed_digest(node):
                return '0' * 64 if getattr(node, 'name', None) == target else original_digest(node)

            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, 'reviewed definition changed or is missing:'):
                    verify_inventory()

    def test_new_complete_modules_reject_an_unreviewed_top_level_statement(self) -> None:
        original_parse = ast.parse

        def parse_with_extra_statement(source, filename='<unknown>', *args, **kwargs):
            tree = original_parse(source, filename, *args, **kwargs)
            if str(filename).replace('\\', '/').endswith(f'/{module}.py'):
                tree.body.append(ast.Assign(
                    targets=[ast.Name(id='UNREVIEWED_POLICY', ctx=ast.Store())],
                    value=ast.Constant(value=True),
                ))
            return tree

        for module in ('spherical_projection_bounds', 'spherical_preflight',
                       'geometry_quality', 'spherical_projection_cpu', 'spherical_sampling_cuda'):
            with self.subTest(module=module), mock.patch.object(inventory.ast, 'parse', side_effect=parse_with_extra_statement):
                with self.assertRaisesRegex(RuntimeError, f'complete-module statement coverage differs: {module}'):
                    verify_inventory()

    def test_patch_review_keeps_the_prior_release_authenticated(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        prior = inventory.reviewed_v21_contract(manifest)
        patch = inventory.reviewed_v21_patch_contract(manifest, prior)
        self.assertEqual(prior['release'], '21.0.0')
        self.assertEqual(patch['release'], '21.0.1')
        self.assertEqual(patch['previous_review_sha256'], inventory.REVIEWED_V21_SHA256)
        manifest['v21_review']['definitions'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21 review digest mismatch'):
            inventory.reviewed_v21_contract(manifest)

    def test_patch_admission_and_memory_contracts_cannot_change_without_review(self) -> None:
        for module, name in (
            ('backprojection', '_MainProcessGpuStageCoordinator'),
            ('publication_memory', 'native_fullframe_dense_reserve'),
        ):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            record = next(item for item in manifest['v21_0_1_review']['definitions']
                          if (item['module'], item['name']) == (module, name))
            record['sha256'] = '0' * 64
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'v21.0.1 review digest mismatch'):
                inventory.reviewed_v21_patch_contract(manifest, manifest['v21_review'])

    def test_patch_supersessions_chain_from_both_v21_and_v20_pins(self) -> None:
        for module, name in (
            ('pipeline', '_main_impl'),
            ('publication_memory', 'plan_native_publication_memory'),
        ):
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            patch = manifest['v21_0_1_review']
            record = next(item for item in patch['definitions']
                          if (item['module'], item['name']) == (module, name))
            record['previous_sha256'] = '0' * 64
            reauthenticated = hashlib.sha256(json.dumps(
                patch, sort_keys=True, separators=(',', ':'),
            ).encode('utf-8')).hexdigest()
            with self.subTest(name=name), mock.patch.object(inventory, 'REVIEWED_V21_0_1_SHA256', reauthenticated):
                with self.assertRaisesRegex(RuntimeError, f'supersession does not match its historical pin: {module}.{name}'):
                    inventory.reviewed_v21_patch_contract(manifest, manifest['v21_review'])

    def test_patch_checks_current_added_definitions_and_retry_policy(self) -> None:
        original_digest = inventory.digest
        for target, expected_error in (
            ('native_fullframe_dense_reserve', 'reviewed definition changed or is missing: publication_memory.native_fullframe_dense_reserve'),
            ('_CUDA_RECHECK_SLICES', 'reviewed statement changed or is missing: spherical_projection.binding__CUDA_RECHECK_SLICES'),
        ):
            def changed_digest(node):
                names = [getattr(node, 'name', None)]
                if isinstance(node, ast.Assign):
                    names.extend(getattr(item, 'id', None) for item in node.targets)
                return '0' * 64 if target in names else original_digest(node)

            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    verify_inventory()

    def test_numba_compile_policy_is_pinned_outside_function_bodies(self) -> None:
        key = ('cylindrical_projection', 'numba_compile_policy')
        _expected_hash, reason = REVIEWED_V20_ADDED_STATEMENTS[key]
        with mock.patch.dict(REVIEWED_V20_ADDED_STATEMENTS, {key: ('0' * 64, reason)}):
            with self.assertRaisesRegex(RuntimeError, 'reviewed added statement changed or is missing: cylindrical_projection.numba_compile_policy'):
                verify_inventory()

    def test_scalar_cuda_kernel_has_an_independent_review_pin(self) -> None:
        key = ('cuda_backend', '_radial_native_kernels')
        _expected_hash, reason = REVIEWED_V20_ADDED_DEFINITIONS[key]
        with mock.patch.dict(REVIEWED_V20_ADDED_DEFINITIONS, {key: ('0' * 64, reason)}):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.5 supersession does not match its historical pin: cuda_backend._radial_native_kernels'):
                verify_inventory()

        original_digest = inventory.digest

        def changed_kernel(node):
            return '0' * 64 if getattr(node, 'name', None) == key[1] else original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=changed_kernel):
            with self.assertRaisesRegex(RuntimeError, 'v20 reviewed added definition changed or is missing: cuda_backend._radial_native_kernels'):
                verify_inventory()

    def test_azimuthal_rename_pins_exact_ast_without_accepting_shell_names_or_changed_math(self) -> None:
        old = ast.parse('def radial_sample(value):\n    return value + 1\n').body[0]
        renamed = ast.parse('def azimuthal_sample(value):\n    return value + 1\n').body[0]
        altered = ast.parse('def azimuthal_sample(value):\n    return value + 2\n').body[0]
        baseline_hash = digest(old)
        record = {
            'module': 'sample', 'baseline_sha256': baseline_hash,
            'renamed_sha256': digest(renamed), 'baseline_name': 'radial_sample',
            'current_name': 'azimuthal_sample',
        }
        manifest = {
            'statements': [{'module': 'sample', 'sha256': baseline_hash, 'name': 'radial_sample'}],
            'v20_azimuthal_rename': {'statements': [record]},
        }
        replacement = azimuthal_rename_replacements(manifest)['sample', baseline_hash]
        self.assertEqual(replacement, digest(renamed))
        self.assertNotEqual(replacement, digest(old), 'A new shell function reusing the old name is a different declaration')
        self.assertNotEqual(replacement, digest(altered), 'The rename map must not normalize arithmetic changes')
        for updates, message in (
            ({'baseline_sha256': '0' * 64}, 'absent immutable'),
            ({'current_name': 'radial_sample'}, 'identity mismatch'),
            ({'renamed_sha256': 'not-a-digest'}, 'invalid reviewed'),
        ):
            broken = copy.deepcopy(manifest)
            broken['v20_azimuthal_rename']['statements'][0].update(updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(RuntimeError, message):
                azimuthal_rename_replacements(broken)
        duplicated = copy.deepcopy(manifest)
        duplicated['v20_azimuthal_rename']['statements'].append(record)
        with self.assertRaisesRegex(RuntimeError, 'duplicate Azimuthal rename'):
            azimuthal_rename_replacements(duplicated)

    def test_original_inventory_cannot_be_silently_rebaselined(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        manifest['statements'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'inventory.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with mock.patch('tools.verify_package_inventory.MANIFEST', path):
                with self.assertRaisesRegex(RuntimeError, 'immutable inventory digest mismatch'):
                    verify_inventory()

    def test_ast_dump_keeps_empty_fields_across_python_versions(self) -> None:
        function = ast.parse("def callback():\n    pass\n").body[0]

        dumped = stable_ast_dump(function)

        self.assertIn("decorator_list=[]", dumped)

    def test_marker_move_changes_seam_digest_even_when_definition_ast_is_unchanged(self) -> None:
        first_import_reviewed = inspect_seams(
            f"""
            def callback(flag):
                {LOCAL_IMPORT_SEAM_MARKER}
                from .first import run_first
                # The other callback remains local for unrelated reasons.
                from .second import run_second
                return run_first() if flag else run_second()
            """
        )
        second_import_reviewed = inspect_seams(
            f"""
            def callback(flag):
                # The first callback remains local for unrelated reasons.
                from .first import run_first
                {LOCAL_IMPORT_SEAM_MARKER}
                from .second import run_second
                return run_first() if flag else run_second()
            """
        )

        first_definition, first_seam = first_import_reviewed[("sample", "callback")]
        second_definition, second_seam = second_import_reviewed[("sample", "callback")]
        self.assertEqual(first_definition, second_definition)
        self.assertNotEqual(first_seam, second_seam)

    def test_marker_inside_method_pins_the_enclosing_top_level_class(self) -> None:
        reviewed = inspect_seams(
            f"""
            class CallbackOwner:
                def callback(self):
                    {LOCAL_IMPORT_SEAM_MARKER}
                    from .dependency import run
                    return run()
            """
        )
        self.assertEqual(set(reviewed), {("sample", "CallbackOwner")})

    def test_marker_must_immediately_precede_a_relative_function_local_import(self) -> None:
        invalid_sources = (
            f"""
            def callback():
                {LOCAL_IMPORT_SEAM_MARKER}

                from .dependency import run
            """,
            f"""
            def callback():
                {LOCAL_IMPORT_SEAM_MARKER}
                from dependency import run
            """,
            f"""
            {LOCAL_IMPORT_SEAM_MARKER}
            from .dependency import run
            """,
        )
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                inspect_seams(source)

    def test_changed_binding_review_must_reference_the_immutable_inventory(self) -> None:
        unknown = ("config", "0" * 64)
        with mock.patch.dict(
            INTENTIONALLY_CHANGED_BINDINGS,
            {unknown: "SAVE_OPTION_TOKENS"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "changed-binding entries are absent from the immutable inventory",
            ):
                verify_inventory()


if __name__ == "__main__":
    unittest.main()
