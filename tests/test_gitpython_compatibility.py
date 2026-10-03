"""Exercise the real WandB Git client and repository discovery security boundary."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import git
from wandb.sdk.lib.gitlib import GitRepo


class GitCompatibility(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo_dir = self.root / "source"
        self.repo_dir.mkdir()
        # Avoid user/global configuration, hooks and credentials in all fixtures.
        self.old_env = os.environ.copy()
        self.addCleanup(self.restore_env)
        os.environ.update({
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "Compatibility Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Compatibility Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
        })
        self.command("init", "-b", "main")
        self.command("config", "user.email", "fixture@example.invalid")
        self.command("remote", "add", "origin", "https://example.invalid/fixture.git")
        (self.repo_dir / "train.py").write_text("loss = 1.0\n")
        self.command("add", "train.py")
        self.command("commit", "-m", "initial fixture")

    def restore_env(self):
        os.environ.clear()
        os.environ.update(self.old_env)

    def command(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo_dir), *args], text=True).strip()

    def test_wandb_metadata_and_changes(self):
        tracker = GitRepo(str(self.repo_dir), lazy=False)
        self.assertTrue(tracker.enabled)
        self.assertEqual(Path(tracker.root_dir).resolve(), self.repo_dir.resolve())
        self.assertEqual(tracker.last_commit, self.command("rev-parse", "HEAD"))
        self.assertEqual(tracker.branch, "main")
        self.assertEqual(tracker.email, "fixture@example.invalid")
        self.assertEqual(tracker.remote_url, "https://example.invalid/fixture.git")
        self.assertFalse(tracker.dirty)
        (self.repo_dir / "train.py").write_text("loss = 0.5\n")
        (self.repo_dir / "untracked.txt").write_text("fixture\n")
        self.assertTrue(tracker.dirty)
        self.assertTrue(tracker.is_untracked("untracked.txt"))
        self.assertIn("+loss = 0.5", tracker.run_git("diff"))
        self.assertFalse(tracker.is_untracked("train.py"))
        self.assertIsNone(tracker.get_upstream_fork_point())

    def test_local_clone_bare_repo_and_linked_worktree(self):
        source = git.Repo(self.repo_dir)
        clone = git.Repo.clone_from(str(self.repo_dir), self.root / "clone")
        self.assertEqual(clone.head.commit.hexsha, source.head.commit.hexsha)
        self.assertEqual(clone.head.commit.tree["train.py"].data_stream.read(), b"loss = 1.0\n")
        bare = git.Repo.clone_from(str(self.repo_dir), self.root / "bare", bare=True)
        self.assertTrue(bare.bare)
        self.assertEqual(bare.head.commit.hexsha, source.head.commit.hexsha)
        worktree = self.root / "linked"
        self.command("worktree", "add", "-b", "linked", str(worktree))
        linked = GitRepo(str(worktree), lazy=False)
        self.assertEqual(linked.last_commit, source.head.commit.hexsha)
        self.assertEqual(linked.branch, "linked")
        self.assertEqual(Path(linked.root_dir).resolve(), worktree.resolve())

    @unittest.skipIf(os.environ.get("GITPYTHON_BASELINE") == "1", "pre-fix discovery is intentionally unsafe")
    def test_tracked_content_cannot_shadow_real_git_directory(self):
        # GHSA-239g-whfq-7xj9: only inert tracked files, never an executable hook.
        (self.repo_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (self.repo_dir / "gitdir").write_text(".git\n")
        (self.repo_dir / "commondir").write_text(".git\n")
        (self.repo_dir / "config").write_text("[user]\n\temail = shadow@example.invalid\n")
        for folder in ("objects", "refs"):
            (self.repo_dir / folder).mkdir()
            (self.repo_dir / folder / "fixture.txt").write_text("inert tracked content\n")
        self.command("add", "HEAD", "gitdir", "commondir", "config", "objects", "refs")
        self.command("commit", "-m", "inert shadow fixture")
        repo = git.Repo(self.repo_dir)
        self.assertEqual(Path(repo.git_dir).resolve(), (self.repo_dir / ".git").resolve())
        self.assertEqual(repo.config_reader().get_value("user", "email"), "fixture@example.invalid")
        tracker = GitRepo(str(self.repo_dir), lazy=False)
        self.assertEqual(tracker.email, "fixture@example.invalid")
        clone = git.Repo.clone_from(str(self.repo_dir), self.root / "shadow-clone")
        self.assertEqual(Path(clone.git_dir).resolve(), (self.root / "shadow-clone" / ".git").resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
