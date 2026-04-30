"""Smoke tests for BalaRL - verify all subsystems and environment."""

import sys
import time


def test_env_import():
    from balarl.engine.cards import Card, Suit, Rank, create_standard_deck
    from balarl.engine.scoring import ScoreEngine, HandType
    from balarl.engine.hand_eval import classify_hand
    from balarl.engine.jokers import JokerInfo, JOKER_LIBRARY, JOKER_ID_TO_INFO
    from balarl.engine.joker_effects import JokerEffects
    from balarl.engine.shop import Shop, PlayerState
    from balarl.engine.game_state import UnifiedGameState
    from balarl.engine.blinds import get_blind_chips, get_blind_reward
    from balarl.engine.boss_blinds import BossBlindManager
    from balarl.engine.consumables import ConsumableManager, TAROT_NAMES, PLANET_NAMES
    from balarl.env.balatro_env import BalatroEnv
    print("  PASS: All modules imported")


def test_cards():
    from balarl.engine.cards import Card, Suit, Rank, create_standard_deck
    deck = create_standard_deck()
    assert len(deck) == 52
    ace = Card(Rank.ACE, Suit.SPADES)
    assert ace.is_face is False
    king = Card(Rank.KING, Suit.HEARTS)
    assert king.is_face is True
    assert king.base_chips == 10
    for c in deck:
        assert Card.from_id(c.card_id) == c
    print("  PASS: Card primitives")


def test_scoring():
    from balarl.engine.scoring import ScoreEngine, HandType
    se = ScoreEngine()
    chips, mult = se.get_base_chips_mult(HandType.FLUSH)
    assert chips == 35 and mult == 4
    se.apply_planet(HandType.FLUSH)
    chips, mult = se.get_base_chips_mult(HandType.FLUSH)
    assert chips == 45 and mult == 5
    print("  PASS: Scoring engine")


def test_hand_eval():
    from balarl.engine.cards import Card, Suit, Rank
    from balarl.engine.hand_eval import classify_hand
    from balarl.engine.scoring import HandType

    flush = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.JACK, Suit.HEARTS), Card(Rank.TEN, Suit.HEARTS)]
    ht, _ = classify_hand(flush)
    assert ht == HandType.STRAIGHT_FLUSH, f"Got {ht.name}"

    pair = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.DIAMONDS), Card(Rank.QUEEN, Suit.CLUBS), Card(Rank.THREE, Suit.SPADES)]
    ht, _ = classify_hand(pair)
    assert ht == HandType.ONE_PAIR, f"Got {ht.name}"

    quads = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.ACE, Suit.CLUBS), Card(Rank.THREE, Suit.SPADES)]
    ht, _ = classify_hand(quads)
    assert ht == HandType.FOUR_KIND, f"Got {ht.name}"

    print("  PASS: Hand evaluation")


def test_joker_library():
    from balarl.engine.jokers import JOKER_LIBRARY, JOKER_ID_TO_INFO, JOKER_NAME_TO_ID
    assert len(JOKER_LIBRARY) == 150
    for name in ["Triboulet", "Blueprint", "Perkeo", "Baron", "Mime"]:
        assert name in JOKER_NAME_TO_ID, f"Missing {name}"
    print("  PASS: Joker library (150)")


def test_joker_effects():
    from balarl.engine.jokers import JOKER_NAME_TO_ID, JOKER_ID_TO_INFO
    from balarl.engine.joker_effects import JokerEffects
    from balarl.engine.cards import Card, Suit, Rank

    je = JokerEffects(seed=42)
    joker = JOKER_ID_TO_INFO[JOKER_NAME_TO_ID["Joker"]]
    effect = je.apply(joker, "scoring", {}, {})
    assert effect and effect.get("mult") == 4

    scary = JOKER_ID_TO_INFO[JOKER_NAME_TO_ID["Scary Face"]]
    effect = je.apply(scary, "individual_scoring", {"card": Card(Rank.KING, Suit.HEARTS)}, {})
    assert effect and effect.get("chips") == 30

    fib = JOKER_ID_TO_INFO[JOKER_NAME_TO_ID["Fibonacci"]]
    effect = je.apply(fib, "individual_scoring", {"card": Card(Rank.ACE, Suit.SPADES)}, {})
    assert effect and effect.get("mult") == 8

    print("  PASS: Joker effects")


def test_blinds():
    from balarl.engine.blinds import get_blind_chips
    assert get_blind_chips(1, "Small Blind") == 300
    assert get_blind_chips(8, "Boss Blind") == 10500
    ante9 = get_blind_chips(9, "Small Blind")
    assert ante9 > 5250, f"Endless scaling broken: {ante9}"
    print("  PASS: Blind scaling")


def test_env_basic():
    from balarl.env.balatro_env import BalatroEnv
    env = BalatroEnv(seed=42)
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "hand" in obs
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < 200:
        mask = obs["action_mask"]
        legal = [i for i, v in enumerate(mask) if v > 0]
        if not legal:
            break
        action = legal[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    print(f"  PASS: Env basic run: {steps} steps, ante={int(obs['ante'])}")


def test_env_expert():
    from balarl.env.balatro_env import BalatroEnv
    from balarl.agents.expert import ExpertAgent, run_expert_episode
    env = BalatroEnv(seed=3)
    agent = ExpertAgent(seed=1003)
    reward, ante, steps = run_expert_episode(env, agent, max_steps=2000)
    assert ante >= 3, f"Expert should reach ante >= 3, got ante={ante}"
    print(f"  PASS: Expert agent: ante={ante}, steps={steps}, reward={reward:.1f}")


def main():
    print("=" * 60)
    print("BalaRL Smoke Tests")
    print("=" * 60)

    tests = [
        ("Module imports", test_env_import),
        ("Card primitives", test_cards),
        ("Scoring engine", test_scoring),
        ("Hand evaluation", test_hand_eval),
        ("Joker library", test_joker_library),
        ("Joker effects", test_joker_effects),
        ("Blind scaling", test_blinds),
        ("Environment basic", test_env_basic),
        ("Expert agent", test_env_expert),
    ]

    passed = 0
    failed = 0
    t0 = time.perf_counter()

    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
