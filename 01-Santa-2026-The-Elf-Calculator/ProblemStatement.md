## Santa 2026 - The Elf Calculator
Write arithmetic programs to build large target numbers from small starting values in the fewest steps

### Overview
Santa's workshop has a calculator problem. Each toy-making station needs to produce a list of exact numbers, but its Elf Calculator can store only 48 values at a time. Every instruction adds, subtracts, multiplies, or exactly divides two stored values and replaces the value in one storage slot.

Your task is to write the shortest programs you can for 1,000 puzzles.

### Description
Santa and Rudolph solving arithmetic puzzles on a sunny summer day by the beachThe workshop's calculator caught a Christmas chill,
With giant target numbers left for elves to fill!
They only have a tiny starting seed in hand,
And four simple symbols at the sleigh's command.

Add, subtract, divide, or multiply with care,
Save each new result with clever steps to spare!
From a humble three, watch the totals grow,
Building every target in a tidy row.

Every extra step slows the sleigh's ascent,
So make each calculation short and well-spent.
Keep your program brief, make your logic shine,
And help Santa launch right on Christmas time!

### The Challenge

The North Pole's toy-counting calculator is stuck in basic mode just as the holiday rush hits its peak! It can only perform one simple math operation at a time, and every extra step slows down the assembly line. 
> **In this competition, you’ll write programs that use basic arithmetic (+, -, ×, ÷) to generate specific large target numbers from small initial values using as few total instructions as possible.**

### The Goal

> **Your score is the total number of arithmetic steps your programs use across all puzzles**—so every instruction you shave off moves you higher on the leaderboard. The teams that craft the shortest, most efficient solutions will win Rudolph's heart and claim a share of the prize pool.

How far can you go with just four basic operations? Sharpen your algorithms, trim every extra step, and help keep Santa's workshop running on time!

## Evaluation
### Scoring
Submissions are evaluated by replaying the arithmetic program for each of the 1,000 scored puzzles. Your score is the total number of instructions across all 1,000 programs. Lower is better. There is no public/private leaderboard split.


Every instruction costs one, including instructions that use the same source twice, produce a value you have already calculated, or run after all targets have been covered. All instructions must be legal. There is no partial credit. Each program must follow the calculator rules on the Data page and produce every target at least once. If any program is invalid, the submission is rejected.

### Submission format
Submit a UTF-8 CSV file with exactly two columns, id and program, in that order. Include every ID from scored_0001 through scored_1000 exactly once. Row order does not affect the score.

Each program cell contains a JSON array of instructions. Each instruction is a four-element array:

```
["+", 0, 1, 2]
```

This example adds the values in registers 0 and 1 and writes the result to register 2.

- The operator must be one of "+", "-", "*", or "/".
- The next two elements are integer source register indices from 0 through 47. Both registers must contain defined values.
- The final element is the integer destination register index, also from 0 through 47.
- Both source values are read before the destination is overwritten. The destination may be the same register as either source, and both sources may be the same register.

The following CSV shows a complete solution for the toy puzzle on the Data page. Its ID is illustrative and must not be included in a competition submission. Use sample_submission.csv for a complete submission with the actual scored IDs.

```
id,program
example_0001,"[[""/"",0,0,1],[""-"",0,1,2],[""+"",0,2,3],[""+"",0,3,4]]"
```
Use a CSV writer to escape the quotation marks inside each JSON cell.

### Program limits

| Parameter                    | Limit                                                        |
|------------------------------|--------------------------------------------------------------|
| Registers per puzzle         | 48, indexed from 0 through 47                                |
| Instructions per program     | 4,096                                                        |
| UTF-8 bytes per program cell | 131,072 bytes (128 KiB)                                      |
| Value of any register        | A positive integer no greater than that puzzle's value_limit |

Every puzzle has a value_limit no greater than 10^18. The sample submission uses at most 2,873 instructions for any puzzle.

---
## Dataset Description
### Files
puzzles.csv - Contains the 1,000 scored puzzles. Each row is a separate puzzle.

id - unique puzzle identifier, from scored_0001 through scored_1000.
starting_values - JSON array of starting positive integers. Values occupy registers 0 through len(starting_values) - 1 at the start.
targets - JSON array of positive integers the program must produce at least once, in increasing order.
value_limit - largest value allowed in any register during execution. This is an integer between 1 and 10^18.
registers - number of available registers. This is 48 for every puzzle.
practice_puzzles.csv - Contains 200 unscored puzzles, with IDs from practice_0001 through practice_0200. It has the same five puzzle columns as puzzles.csv, plus:

family - generator family: independent, shared, multiplicative, mixed, or tiny.
design_tier - generator difficulty tier: hard, medium, or easy.
lower_bound - number of targets not already present in starting_values, calculated as len(set(targets) - set(starting_values)). A single instruction can produce at most one new target, so this is a lower bound on the instruction count. It is generally loose and is not guaranteed to be achievable.

The practice set uses the same puzzle families and arithmetic rules as the scored set, with a larger share of tiny puzzles for debugging. Family proportions differ between the two sets.

sample_submission.csv - Contains a valid baseline program for every puzzle.

### Calculator rules
Each puzzle has 48 registers, indexed from 0 through 47. At the start, the values in starting_values occupy the first registers in their listed order. All other registers are undefined.

An instruction ["op", i, j, d] reads source registers i and j, applies the operator, and writes the result to destination register d. Writing to a register replaces its previous value. **Both operands are read before the write, so the destination may be one of the source registers.**

- Values must be positive integers. Subtraction is legal only when the first operand is greater than the second.
- Division must be exact. The second operand must divide the first with no remainder.
- Every result must be at most value_limit. This includes intermediate values, even if a later instruction would reduce them.
- Source registers must be defined. An undefined register becomes defined after an instruction writes to it.
- Targets only need to appear once. A target is covered when its value appears in any register, including at the start. It remains covered after that register is overwritten.

Repeated source operands are allowed. For example, ["+", 3, 3, 7] doubles the value in register 3, and ["/", 0, 0, 4] produces 1 from any positive value in register 0. Initial registers may also be overwritten.

All instructions count toward the score, including those after the last target has been produced. The Evaluation page lists the instruction and cell-size limits.

### Worked example
This small example illustrates the file format and rules. It is not one of the scored puzzles.
```
id,starting_values,targets,value_limit,registers
example_0001,[3],"[1, 2, 5, 8]",9,48
```
**Box 0 holds 3. You must produce 1, 2, 5 and 8, and nothing may ever exceed 9.**

The following program covers all four targets in four instructions:
```
[["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]
```
| Step | Calculation | Destination | Covered targets |
|------|-------------|-------------|-----------------|
| 1    | 3 / 3 = 1   | Register 1  | 1               |
| 2    | 3 - 1 = 2   | Register 2  | 1, 2            |
| 3    | 3 + 2 = 5   | Register 3  | 1, 2, 5         |
| 4    | 3 + 5 = 8   | Register 4  | 1, 2, 5, 8      |


## Initial Thoughts by Me

- There are 1000 puzzles(analogous to samples in testset)
- Elf Calculator is like the memory of microprocessoor with 48 memory slots each of 128KB.
- Basic Action/instruction is [Operator, First_Memory_Address, Second_Memory_Address, Target_Memory_Address], $operator \in [+,-,*,/]$ and the result is noted in target address, so no calculating once and pasting multiple times to save instructions/actions
- Target_Memory_Address may be one of the source addresses
- Max 4096 instructions per puzzle/sample
- submission.csv has 2 columns: id and program, where $id \in [scored0001, scored1000]$, program is the action in quotes.
- Dataset ```puzzles.csv``` has 1000 **scored** rows(samples/puzzles) and 5 columns: id, starting_value, targets, value_limit, registers
- Dataset ```practice_puzzles.csv``` has 200 **unscored** rows(samples/puzzles) and 8 columns: same_5_columns_as_in_puzzles.csv, family, design_tier, lower_bound

> $family \in [independent, shared, multiplicative, mixed, tiny]$
> $design_tier  \in [hard, medium, easy]$

- Instructions from ```puzzles.csv``` are scored whereas ```practice_puzzles.csv``` contains extra columns to focus on which types of puzzles our solution needs improvement on(extra columns are like puzzle categories so that we can improve the solution to perform better on certain types/categories of puzzles)

### Example Explained
```
id,starting_values,targets,value_limit,registers
example_0001,[3],"[1, 2, 5, 8]",9,48
```
**Box 0 holds 3. You must produce 1, 2, 5 and 8, and nothing may ever exceed 9.**

| Instruction |            Meaning            | Boxes afterwards | Targets done |
|:-----------:|:-----------------------------:|:----------------:|:------------:|
| ["/",0,0,1] | box0 ÷ box0 = 3÷3 = 1 → box 1 | 3, 1             | 1            |
| ["-",0,1,2] | box0 − box1 = 3−1 = 2 → box 2 | 3, 1, 2          | 1, 2         |
| ["+",0,2,3] | box0 + box2 = 3+2 = 5 → box 3 | 3, 1, 2, 5       | 1, 2, 5      |
| ["+",0,3,4] | box0 + box3 = 3+5 = 8 → box 4 | 3, 1, 2, 5, 8    | all four     |

That’s 4 instructions, so the score is 4. An instruction is always [operation, first box, second box, box to write into]. Something like ["*",3,3,5] (5×5 = 25) would be illegal here, because 25 > 9.

So, the sample submission should be "[["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]"

