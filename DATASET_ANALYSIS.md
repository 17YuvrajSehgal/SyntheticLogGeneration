# Dataset Analysis Results

## Training Data Statistics (compress-gzip, sequence_length_200)

### Overview
- **Total sequences:** 10,000
- **Sequence length:** Fixed at 200 events (all sequences are exactly 200)
- **Total events:** 2,000,000
- **Unique event IDs:** 62
- **Event ID range:** 1 to 62 (⚠️ **Note: starts at 1, not 0**)

### Event Distribution
The top 10 events account for a significant portion of the data:
- **Event 1:** 19.99% (most frequent)
- **Event 2:** 19.55%
- **Event 3:** 19.25%
- **Event 4:** 12.29%
- **Event 5:** 6.25%
- **Event 6:** 4.05%
- **Event 7:** 3.35%
- **Event 8:** 2.30%
- **Event 9:** 1.96%
- **Event 10:** 1.83%

**Top 3 events account for ~59% of all events** - this is important for the model to learn.

### Key Observations

1. **Event ID Range:** Events range from 1-62 (not 0-61)
   - This is important for generation - we must ensure generated IDs are in this range
   - The fix in `generate.py` should handle this correctly

2. **Uniform Sequence Length:** All sequences are exactly 200 events
   - This simplifies training and generation
   - No need for padding or variable-length handling

3. **Skewed Distribution:** Top 3 events are very common (~60% combined)
   - Model should learn to generate these frequently
   - Less common events (events 5-62) should appear less often

4. **Vocabulary Size:** 62 unique events
   - This is a manageable vocabulary size
   - Model should be able to learn all event types

### Comparison with Testing Data

**Important Note:** Earlier evaluation showed testing data has only 32 unique events, while training has 62. This suggests:
- Testing data might be a subset or different distribution
- We should verify this discrepancy
- Generation should be constrained to events that exist in **both** training and testing

### Implications for Generation

1. **Valid Event IDs:** Generated sequences should only contain IDs 1-62
2. **Distribution:** Generated sequences should reflect the skewed distribution (top 3 events ~60%)
3. **Sequence Length:** All generated sequences should be exactly 200 events
4. **Vocabulary Constraint:** The model should learn to generate only valid event IDs

### Next Steps

1. ✅ Dataset analyzed - confirmed 62 unique events (1-62)
2. ⏭️ Verify testing data statistics match
3. ⏭️ Ensure generation code respects the 1-62 range
4. ⏭️ Train model and check if it learns the distribution
5. ⏭️ Evaluate if generated sequences match the event frequency distribution

