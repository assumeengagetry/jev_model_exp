import unittest
from benchmark import aggregate, encode_pair, fixed_padding_stats, profile_stats, rounded


class BenchmarkTests(unittest.TestCase):
    def test_padding_uses_each_physical_microbatch(self):
        rows = [{"input_ids": list(range(n))} for n in (9, 16, 17, 22)]
        stats = profile_stats(rows, 2)
        self.assertEqual([batch["shape"] for batch in stats["microbatches"]], [[2, 16], [2, 24]])
        self.assertEqual(stats["padded_tokens"], 80)
        self.assertEqual(stats["nonpadding_tokens"], 64)
        self.assertAlmostEqual(stats["padding_fraction"], .2)

    def test_exact_lengths_keep_hypothesis_and_eos(self):
        pair = {"premise_ids": list(range(100)), "hypothesis_ids": [201, 202], "label": 1}
        for length in (8, 9, 16, 24):
            row = encode_pair(pair, length, [301], [302], 303)
            self.assertEqual(len(row["input_ids"]), length)
            self.assertEqual(row["input_ids"][-4:], [302, 201, 202, 303])
            self.assertEqual(row["attention_mask"], [1] * length)
            self.assertEqual(row["labels"], 1)
        with self.assertRaises(ValueError):
            encode_pair(pair, 3, [301], [302], 303)
        with self.assertRaises(ValueError):
            encode_pair(pair, 256, [301], [302], 303)

    def test_aggregate_uses_tokens_over_summed_seconds(self):
        rows = [{"state": "complete", "seconds": seconds, "input": {"nonpadding_tokens": 100},
                 "peak_allocated_GiB": 1, "peak_reserved_GiB": 2,
                 "forward_seconds": .2, "backward_seconds": .8} for seconds in (1, 2)]
        value = aggregate(rows)
        self.assertEqual(value["tokens_per_second"], 200)
        self.assertEqual(value["successful_repeats"], 2)
        self.assertEqual(aggregate([{"state": "oom"}])["state"], "OOM")

    def test_round_up(self):
        self.assertEqual([rounded(n) for n in (192, 205, 2572, 2807)], [192, 208, 2576, 2808])

    def test_fixed_padding_records_actual_tensor_shapes(self):
        rows = [{"input_ids": list(range(n))} for n in (9, 16, 17, 22)]
        stats = fixed_padding_stats(profile_stats(rows, 2), 32)
        self.assertEqual(stats["padded_tokens"], 128)
        self.assertEqual([b["shape"] for b in stats["microbatches"]], [[2, 32], [2, 32]])
        self.assertEqual(stats["padding_fraction"], .5)


if __name__ == "__main__":
    unittest.main()
