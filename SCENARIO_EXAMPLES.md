# Scenario Examples (Stepwise Lava Mode)

## Scenario A: Learning Trigger

1. Agent explores Lava room.
2. Agent dies first time.
3. Agent respawns and dies second time.
4. Failure snapshot is written.
5. Rule is generated and verified.

## Scenario B: Truth Confirmation

1. Same Lava room opens again.
2. Explorer checks front tile each step.
3. If `red lava` matches memory rule, explorer turns.
4. Explorer reaches goal without hazard.
5. Repeat for 3 confirmation episodes.

## Scenario C: Stop Point (Current)

- Run ends after Lava confirmation block.
- Sand and Final Exam are paused for now.
