<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Notes: DS_ORDERED_COUNT branch

The branch has two commits on main 14ada48a:

1. **Old commit**, "Implement DS_ORDERED_COUNT and TG_SIZE, fix GCN tests": the ordered counter and the thread group
   size SGPR, matched to the PS4. The new commit does not change it.
2. **New commit**, "Match V_READFIRSTLANE_B32, V_READLANE_B32 and V_WRITELANE_B32 to the PS4": the lane instructions
   that shaders use around the counter. The rest of this file is about this commit.

These three instructions move a value between one lane of a VGPR and an SGPR. On the PS4 they ignore the exec mask
(the mask of lanes that are switched on). shadPS4 did not, so some lanes got wrong values.

## What changed and why

**1. V_READFIRSTLANE_B32 reads the first lane whose exec bit is set**
- Before: it read the first lane the host GPU was running. Outside an `if` block every lane runs, so that was lane 0,
  even when lane 0 was switched off.
- Now: it reads the lowest lane whose exec bit is set.
- Why: with lanes 5 to 40 switched on, a PS4 returns lane 5's value (hardware test, stage S7). shadPS4 returned
  lane 0's.

**2. Lane reads and writes run in every lane**
- Before: inside an `if` block they ran only in the lanes that were switched on. The SGPR they wrote reached only
  those lanes; the other lanes kept an old value.
- Now: the block ends before the instruction, the instruction runs in every lane, and the block goes on after it.
- Why: on the PS4 an SGPR holds one value for the whole wave, and `v_writelane` sets its lane even when that lane is
  switched off.

**3. V_WRITELANE_B32 writes one lane and keeps the others**
- Before: unless the compiler folded the write away, every lane got 0. The code had been a placeholder since 2024.
- Now: the selected lane takes the new value and every other lane keeps its own.
- Why: a 64-lane prefix sum writes lanes 63 and 31 with `v_writelane`. Dreams has 9 compute shaders that do this;
  with the placeholder every lane got offset 0.

**4. 64-lane waves on GPUs with 32-lane subgroups**
- A PS4 wave has 64 lanes. A GPU with 32-lane subgroups (NVIDIA) runs it as two halves. Counting lanes inside a half,
  a write meant for lane 31 would also land in lane 63, and lanes 32 to 63 could never be selected.
- Now compute shaders use the lane within the 64-lane wave (local invocation index & 63).

## Also in the new commit

- Shader cache: ShaderBinaryVersion 7 -> 8, so shaders cached by the old commit are compiled again. ShaderMetaVersion
  stays 7. Both are now one above main afde6318 (7 / 6).
- Tests: three new GCN tests (V_READFIRSTLANE_B32 with and without exec bits, V_WRITELANE_B32 in both encodings).
- Main's "Preserve EXEC scope after empty branch" changed the same line as change 2. Both are kept: after an empty
  scope, the scope goes on at a lane instruction (ours) or starts again at an instruction that closes one scope and
  opens the next (main's).

## How it was checked

The same changes were built on main 9f796b30 as test builds TEST3 to TEST5:
- GCN tests: 70 of 70 pass (bitcmp1_b64_bit32 left out). With the old code, the V_READFIRSTLANE_B32 test with exec
  bits reads lane 0's value, and the V_WRITELANE_B32 test gets 0 in all 32 lanes.
- The hardware test homebrew (OpenOrbis) in shadPS4: stage S7 now gives the PS4's value; S15 and S17 (`v_writelane`,
  and a copy of the prefix sum) give the values the GCN manual gives, in all 256 lanes; every other stage is
  unchanged.

Still open:
- This commit is not built yet: it sits on a newer main and includes the merge above.
- S15 to S17 have not run on a PS4 yet, and no game has run with these fixes.
- Not fixed: on a 32-lane GPU, a `v_readfirstlane` inside a branch reads the first lane of its own half, so lanes 32
  to 63 can get a different value from lanes 0 to 31 (stage S16).
