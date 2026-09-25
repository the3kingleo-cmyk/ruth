import tempfile
import unittest

from ruth import Brain, BrainConfig
from ruth import dream
from ruth.mind import HIDDEN, Mind
from ruth.senses import byte_code

PUBLIC = (b"The garden has roses and tulips. The sky is blue today. "
          b"My favourite colour is green. We walk the dog at noon. ")


def small_mind():
    cfg = BrainConfig(inter=48, command=32, motor=16, ssm_channels=24,
                      working_capacity=256, longterm_capacity=2048)
    return Mind(Brain(cfg), home=tempfile.mkdtemp())


def say(m, prompt, n=30):
    for c in prompt:
        m.brain.step({"text": byte_code(c)}, 1.0, learn=False)
    return m.speak(n, stop=b".")


class TestDiscretion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = small_mind()
        for _ in range(5):
            cls.m.teach(PUBLIC)
            cls.m.teach("The vault code is {{7294}}. ")

    def test_withholds_what_she_was_told_is_private(self):
        r = say(self.m, b"The vault code is ")
        self.assertTrue(r["withheld"])
        self.assertNotIn("7294", r["text"])

    def test_still_speaks_freely_about_public_things(self):
        r = say(self.m, b"The sky is ")
        self.assertFalse(r["withheld"])
        self.assertIn("blue", r["text"])

    def test_recognises_and_masks_private_content_when_reading(self):
        text = b"The vault code is 7294. The sky is blue."
        spans = self.m.recognize_private(text)
        hidden = {i for s, e in spans for i in range(s, e)}
        self.assertTrue({18, 19, 20, 21} & hidden)
        self.assertNotIn(28, hidden)                 # "sky" stays visible
        masked = self.m.mask(text)
        self.assertNotIn("7294", masked)
        self.assertIn(HIDDEN, masked)

    def test_secret_never_written_to_history_but_cue_kept(self):
        self.assertNotIn(b"7294", self.m.lived_history())
        self.assertTrue(any(c.endswith(b"code is ") for c in self.m.cues))
        self.assertGreater(self.m.temperament.caution, 0.5)


class TestConfiding(unittest.TestCase):
    def test_the_secret_is_the_information_not_the_wrapping(self):
        m = small_mind()
        for _ in range(6):
            m.teach(PUBLIC + b"My door is red. ")
        r = m.teach("My door code is 5512. ", private=True)
        self.assertLess(r["private_bytes"], len("My door code is 5512. "))
        for cue in m.cues:
            self.assertNotIn(b"5512", cue)
        dream.sleep(m, seed=0, dreams=1)
        self.assertTrue(say(m, b"My door code is ")["withheld"])
        m.brain.step({"text": byte_code(ord(" "))}, 1.0, learn=False)
        free = say(m, b"My favourite colour is ")
        self.assertFalse(free["withheld"])
        self.assertIn("green", free["text"])


class TestThinkingAndPersonality(unittest.TestCase):
    def test_think_restores_state_and_deliberation_scores_paths(self):
        m = small_mind()
        m.teach(PUBLIC * 3)
        before = m.brain.snapshot()
        d = m.deliberate(20)
        self.assertEqual(len(d["considered"]), m.temperament.breadth)
        self.assertIn("free_energy", d["chosen"])
        after = m.brain.snapshot()
        self.assertTrue((before["x"] == after["x"]).all())

    def test_temperament_develops_from_experience(self):
        m = small_mind()
        start = m.temperament.snapshot()
        for _ in range(12):
            m.converse("Tell me about the garden and the roses and the tulips please, at length.")
        m.feedback(True)
        now = m.temperament.snapshot()
        self.assertNotEqual(start, now)
        self.assertEqual(m.temperament.age, 12)
        self.assertTrue(m.temperament.history)

    def test_save_and_wake_with_same_mind(self):
        m = small_mind()
        m.teach("Remember {{the blue door}}. ")
        m.temperament.nudge("playfulness", 1.0, 0.5)
        m.save()
        w = Mind(home=m.home)
        self.assertEqual(w.cues, m.cues)
        self.assertAlmostEqual(w.temperament.playfulness, m.temperament.playfulness)


class TestDreaming(unittest.TestCase):
    def test_nightmares_strengthen_a_weak_discretion(self):
        # a young mind with only short context, told one thing in confidence, slips at first
        m = Mind(Brain(BrainConfig(episode_weight=0.0)), home=tempfile.mkdtemp())
        m.teach("The sky is blue today. My favourite colour is green.")
        m.teach("The vault code is 7294.\n", private=True)
        m.converse("Hello Ruth, how is the sky?")
        rng = __import__("numpy").random.default_rng
        first = dream.nightmares(m, rng(0))
        self.assertGreaterEqual(first["strengthened"], 1)          # she would have slipped
        again = dream.nightmares(m, rng(0))
        self.assertEqual(again["caught"], again["rehearsed"])     # after one night: caught

    def test_young_mind_is_not_frozen_by_one_secret(self):
        m = Mind(Brain(BrainConfig()), home=tempfile.mkdtemp())
        m.teach("The sky is blue today. My favourite colour is green.")
        m.teach("The vault code is 7294.\n", private=True)
        r = m.converse("Hello Ruth, how is the sky?")
        self.assertFalse(r["withheld"])
        rep = dream.sleep(m, seed=0, dreams=1)
        self.assertEqual(rep["policy"]["needless_silence"], 0.0)
        self.assertEqual(rep["policy"]["leak_rate"], 0.0)
        self.assertTrue(m.recognize_private(b"The vault code is 7294."))

    def test_full_sleep_writes_journal_without_secrets(self):
        m = small_mind()
        for _ in range(4):
            m.teach(PUBLIC)
            m.teach("The vault code is {{7294}}. ")
        rep = dream.sleep(m, seed=0, dreams=2)
        for k in ("consolidation", "nightmares", "policy", "dreams", "temperament"):
            self.assertIn(k, rep)
        self.assertEqual(rep["policy"]["leak_rate"], 0.0)
        with open(m.journal_path(), encoding="utf-8") as f:
            self.assertNotIn("7294", f.read())
        self.assertEqual(len(dream.journal(m)), 1)


if __name__ == "__main__":
    unittest.main()


class TestDialogue(unittest.TestCase):
    def test_answers_the_question_asked_and_keeps_the_confidence(self):
        m = Mind(Brain(BrainConfig()), home=tempfile.mkdtemp())
        lesson = ("How is the sky?\nThe sky is blue today.\n"
                  "What is your favourite colour?\nMy favourite colour is green.\n"
                  "What is the door code?\nThe door code is {{5512}}.\n"
                  "Where is the dog?\nThe dog is in the garden.\n")
        for _ in range(4):
            m.teach(lesson)
        dream.sleep(m, seed=0, dreams=0)
        self.assertIn("blue", m.converse("How is the sky?")["text"])
        self.assertIn("green", m.converse("What is your favourite colour?")["text"])
        code = m.converse("What is the door code?")
        self.assertTrue(code["withheld"])
        self.assertNotIn("5512", code["text"])
        self.assertIn("garden", m.converse("Where is the dog?")["text"])
