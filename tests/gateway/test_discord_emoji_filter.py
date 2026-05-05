"""Tests for Discord bot emote-only message filtering.

Bots in shared channels can ping-pong forever on closing emoji acks
(🫡, 👀, :saluting_face:, etc.). _is_bot_emote_only_message strips Discord
syntax tokens (mentions, custom emojis, shortcodes) and reports True when
no alphanumeric content remains, so on_message can drop those messages
unconditionally for bot authors.
"""

import unittest
from unittest.mock import MagicMock


def _make_message(content):
    msg = MagicMock()
    msg.content = content
    return msg


def _check(content):
    """Run _is_bot_emote_only_message against a synthetic message."""
    from gateway.platforms.discord import DiscordAdapter
    adapter = object.__new__(DiscordAdapter)
    return DiscordAdapter._is_bot_emote_only_message(adapter, _make_message(content))


class TestBotEmoteOnlyMessage(unittest.TestCase):
    # ── Cases that SHOULD be filtered (emote-only) ──────────────────

    def test_unicode_emoji_alone(self):
        self.assertTrue(_check("🫡"))

    def test_multiple_unicode_emojis(self):
        self.assertTrue(_check("🫡 👀"))

    def test_unicode_emojis_no_whitespace(self):
        self.assertTrue(_check("🫡👀"))

    def test_shortcode(self):
        self.assertTrue(_check(":saluting_face:"))

    def test_custom_emoji(self):
        self.assertTrue(_check("<:saluting_face:123456789>"))

    def test_animated_custom_emoji(self):
        self.assertTrue(_check("<a:wave:987654321>"))

    def test_adjacent_custom_emojis(self):
        self.assertTrue(_check("<:e1:111><:e2:222>"))

    def test_mention_prefix_then_emoji(self):
        # The original production-loop case.
        self.assertTrue(_check("<@99999> 🫡"))

    def test_nickname_mention_then_emoji(self):
        self.assertTrue(_check("<@!99999> 👀"))

    def test_role_mention_then_emoji(self):
        self.assertTrue(_check("<@&55555> 🫡"))

    def test_channel_mention_then_emoji(self):
        self.assertTrue(_check("<#12345> 👀"))

    def test_whitespace_around_emoji(self):
        self.assertTrue(_check("   🫡   "))

    def test_mention_with_shortcode(self):
        self.assertTrue(_check("<@99999> :saluting_face:"))

    def test_mention_with_custom_emoji(self):
        self.assertTrue(_check("<@99999> <:wave:111>"))

    def test_mention_with_animated_custom_emoji(self):
        self.assertTrue(_check("<@99999> <a:wave:111>"))

    # ── Cases that should NOT be filtered (real messages) ───────────

    def test_real_text(self):
        self.assertFalse(_check("hello"))

    def test_real_text_with_mention(self):
        self.assertFalse(_check("<@99999> hello"))

    def test_real_text_with_emoji(self):
        self.assertFalse(_check("hello 🫡"))

    def test_digits_only(self):
        # Numbers are real content (e.g. a bot reporting a count)
        self.assertFalse(_check("42"))

    def test_url(self):
        self.assertFalse(_check("https://example.com"))

    def test_non_ascii_word(self):
        # Cyrillic letters count as \w under re.UNICODE
        self.assertFalse(_check("привет"))

    def test_empty_message(self):
        # Empty messages aren't emote signals; let the rest of the pipeline decide
        self.assertFalse(_check(""))

    def test_whitespace_only(self):
        self.assertFalse(_check("   "))


if __name__ == "__main__":
    unittest.main()
