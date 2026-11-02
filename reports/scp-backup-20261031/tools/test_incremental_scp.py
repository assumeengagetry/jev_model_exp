import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class IncrementalScpTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).with_name('incremental_scp.py')
        self.assertTrue(path.is_file(), 'Incremental SCP entry has not been implemented')
        spec = importlib.util.spec_from_file_location('incremental_scp', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def metadata(self, data=b'abc', relative='scripts/runtime.py'):
        return {'source': '/home/dev/assumeengage/' + relative, 'relative': relative,
                'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'mtime_ns': 1, 'snapshot': relative.endswith('.log')}

    def test_policy_rejects_credentials_escape_weights_and_raw_caches(self):
        m = self.module()
        for path in ['../outside.py', '.secrets/modelscope_token', 'config/env.sh',
                     'private-transfer/jobs.json', '.ssh/id_ed25519',
                     'runs/model/final/adapter_model.safetensors',
                     'cache/pair-training/pairs/source.npy', 'envs/jev/README.md']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                m.validate_source(path)
        self.assertEqual(m.validate_source('scripts/runtime.py'),
                         '/home/dev/assumeengage/scripts/runtime.py')

    def test_probe_returns_hash_without_returning_file_contents(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'scripts/runtime.py'
            source.parent.mkdir(); source.write_bytes(b'abc')
            result = m.probe_file(str(source), root=root)
            self.assertEqual(result['sha256'], 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            self.assertEqual(result['size'], 3)
            self.assertNotIn('data', result)
            self.assertNotIn('content', result)

    def test_probe_rejects_sensitive_payload_and_oversized_small_file(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'scripts/runtime.py'
            source.parent.mkdir(); source.write_bytes(b'abcde')
            with self.assertRaises(ValueError):
                m.probe_file(str(source), root=root, max_bytes=4)
            source.write_text('-----BEGIN PRIVATE KEY-----\nsynthetic-test-only\n')
            with self.assertRaises(ValueError):
                m.probe_file(str(source), root=root)

    def test_copy_preserves_old_work_registers_actual_file_and_skips_duplicate(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); backup = root / 'backup'; original = backup / 'files/scripts/runtime.py'
            original.parent.mkdir(parents=True); original.write_bytes(b'old user work')
            downloaded = root / 'download'; downloaded.write_bytes(b'abc')
            meta = self.metadata()
            record = m.record_copy(backup, downloaded, meta)
            self.assertEqual(original.read_bytes(), b'old user work')
            self.assertEqual(Path(record['local_path']).read_bytes(), b'abc')
            self.assertIn('incremental', Path(record['local_path']).parts)
            completed = json.loads((backup / 'SCP-COMPLETED.json').read_text())
            self.assertEqual(completed['file_count'], 1)
            self.assertTrue(completed['files'][0]['present_on_disk'])
            self.assertTrue(m.already_backed_up(backup, meta))
            ledger = json.loads((backup / 'LEDGER.json').read_text())
            self.assertEqual(ledger['entries'][0]['sha256'], meta['sha256'])

    def test_bad_download_does_not_create_completion_record(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); downloaded = root / 'download'; downloaded.write_bytes(b'wrong')
            backup = root / 'backup'
            with self.assertRaises(ValueError):
                m.record_copy(backup, downloaded, self.metadata())
            self.assertFalse((backup / 'SCP-COMPLETED.json').exists())

    def test_snapshot_is_not_registered_as_final(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); downloaded = root / 'download'; downloaded.write_bytes(b'abc')
            record = m.record_copy(root / 'backup', downloaded, self.metadata(relative='logs/running.log'))
            self.assertTrue(record['nonfinal_snapshot'])
            self.assertEqual(record['backup_status'], 'snapshot_verified')

    def test_download_relative_path_cannot_escape_backup(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); downloaded = root / 'download'; downloaded.write_bytes(b'abc')
            with self.assertRaises(ValueError):
                m.record_copy(root / 'backup', downloaded, self.metadata(relative='../../outside.py'))
            self.assertFalse((root / 'outside.py').exists())

    def test_transport_uses_fresh_connections_and_metadata_only_probe(self):
        m = self.module(); meta = self.metadata()

        def endpoint_requires_fresh_connection(command, **kwargs):
            for required in ['ClearAllForwardings=yes', 'ControlMaster=no', 'ControlPath=none']:
                if required not in command:
                    raise AssertionError('SSH endpoint requires fresh no-forwarding transport')
            self.assertEqual(command[0], 'ssh')
            self.assertEqual(json.loads(kwargs['input']), [meta['source']])
            return subprocess.CompletedProcess(command, 0, json.dumps([meta]), '')

        with patch.object(m.subprocess, 'run', side_effect=endpoint_requires_fresh_connection):
            result = m.remote_probe(['scripts/runtime.py'])
        self.assertEqual(result[0]['sha256'], meta['sha256'])

    def test_help_is_local_and_requires_no_network(self):
        m = self.module()
        result = subprocess.run(['python3', m.__file__, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--source', result.stdout)
        self.assertIn('--dry-run', result.stdout)

    def test_destination_symlink_does_not_create_files_outside_backup(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); backup = root / 'backup'; outside = root / 'outside'
            backup.mkdir(); outside.mkdir(); (backup / 'incremental').symlink_to(outside, target_is_directory=True)
            downloaded = root / 'download'; downloaded.write_bytes(b'abc')
            with self.assertRaises(ValueError):
                m.record_copy(backup, downloaded, self.metadata())
            self.assertEqual(list(outside.iterdir()), [])

    def test_modelscope_credential_literal_is_not_accepted_as_metadata(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'config/settings.json'; source.parent.mkdir()
            source.write_text(json.dumps({'modelscope_api_token': 'SYNTHETIC_TEST_ONLY_NOT_A_CREDENTIAL'}))
            with self.assertRaises(ValueError):
                m.probe_file(str(source), root=root)

    def test_completion_pointer_outside_backup_is_not_trusted(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); backup = root / 'backup'; backup.mkdir()
            outside = root / 'outside'; outside.write_bytes(b'abc')
            meta = self.metadata()
            (backup / 'SCP-COMPLETED.json').write_text(json.dumps({'files': [dict(meta, local_path=str(outside))]}))
            self.assertFalse(m.already_backed_up(backup, meta))

    def test_running_json_output_is_marked_nonfinal_even_when_read_is_stable(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'runs/model/evaluation/results.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'complete': False, 'state': 'running'}))
            self.assertTrue(m.probe_file(str(source), root=root)['snapshot'])


if __name__ == '__main__':
    unittest.main()
