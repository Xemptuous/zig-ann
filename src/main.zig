const std = @import("std");
const Allocator = std.mem.Allocator;
const RndGen = std.rand.DefaultPrng;

const BIAS: comptime_float = 1.0;
const ALPHA: comptime_float = 0.1;

const NUM_HIDDEN: u8 = 4;
const NUM_INPUTS: comptime_int = 2;
const NUM_OUTPUTS: comptime_int = 1;
const NUM_TEST_CASES: comptime_int = 4;

const ERROR_THRESHOLD: comptime_float = 0.001;
const EPOCH_THRESHOLD: comptime_int = 1e4 * (std.math.pow(f32, ALPHA, -1));
const EPOCH_PRINT_THRESHOLD: comptime_int = @intFromFloat(ALPHA * 1e4);

const TestCase = struct {
    inputs: [NUM_INPUTS]f32,
    outputs: [NUM_OUTPUTS]f32,
};

const Neuron = struct {
    weights: [NUM_HIDDEN]f32,
    weight_bias: f32,
    inputs: [NUM_HIDDEN]f32 = undefined,
    output: f32 = undefined,
    error_gradient: f32 = undefined,

    ///Create a Neuron and roll its input weights
    pub fn new(a: Allocator, r: *RndGen) !*Neuron {
        const n = try a.create(Neuron);
        Neuron.roll_weights(n, r);
        return n;
    }

    ///Regenerate weights and weight_bias for this Neuron
    pub fn roll_weights(self: *Neuron, r: *RndGen) void {
        self.weight_bias = r.random().float(f32);
        for (0..NUM_HIDDEN) |i|
            self.weights[i] = r.random().float(f32);
    }

    ///Update based on error gradient
    pub fn update_weights(self: *Neuron) void {
        self.weight_bias += ALPHA * BIAS * self.error_gradient;
        for (0..self.weights.len) |i|
            self.weights[i] += ALPHA * self.inputs[i] * self.error_gradient;
    }

    pub fn __tanh(self: *Neuron) f32 {
        var x = self.weight_bias;
        for (self.inputs, self.weights) |i, w|
            x += i * w;
        return std.math.tanh(x);
    }
};

fn NeuralNetwork(comptime T: type) type {
    return struct {
        const This = @This();
        test_cases: *[NUM_TEST_CASES]TestCase,
        input_layer: []T,
        hidden_layer: []*Neuron,
        output_layer: []*Neuron,
        seed: u64 = undefined,
        runtime_ms: i64 = 0,
        num_epochs: usize = 0,
        test_results: *[NUM_TEST_CASES][NUM_OUTPUTS]T = undefined,
        allocator: Allocator,

        pub fn init(allocator: Allocator, cases: *[NUM_TEST_CASES]TestCase) !This {
            const test_cases = cases;
            const input_layer: []T = try allocator.alloc(T, NUM_INPUTS);
            const hidden_layer = try allocator.alloc(*Neuron, NUM_HIDDEN);
            const output_layer = try allocator.alloc(*Neuron, NUM_OUTPUTS);
            const test_results = try allocator.create([NUM_TEST_CASES][NUM_OUTPUTS]T);
            return This{
                .test_cases = test_cases,
                .input_layer = input_layer,
                .hidden_layer = hidden_layer,
                .output_layer = output_layer,
                .test_results = test_results,
                .allocator = allocator,
            };
        }

        fn deinit(this: *This) void {
            for (this.hidden_layer) |n| this.allocator.destroy(n);
            for (this.output_layer) |n| this.allocator.destroy(n);
            this.allocator.free(this.input_layer);
            this.allocator.free(this.hidden_layer);
            this.allocator.free(this.output_layer);
            this.allocator.destroy(this.test_results);
        }

        fn train(this: *This) !void {
            this.seed = @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
            var rng = RndGen.init(this.seed);

            for (0..NUM_HIDDEN) |i|
                this.hidden_layer[i] = try Neuron.new(this.allocator, &rng);

            for (0..NUM_OUTPUTS) |i|
                this.output_layer[i] = try Neuron.new(this.allocator, &rng);

            // An array of hidden layer outputs to easily set output layer inputs
            const hidden_outputs = try this.allocator.alloc(f32, NUM_HIDDEN);
            defer this.allocator.free(hidden_outputs);

            this.runtime_ms = 0;
            var start: i64 = std.time.milliTimestamp();
            while (true) {
                var errors: f32 = 0.0;
                for (this.test_cases) |case| {
                    for (this.hidden_layer, 0..) |n, i| {
                        for (0..NUM_INPUTS) |j|
                            n.inputs[j] = case.inputs[j];
                        n.output = n.__tanh();
                        hidden_outputs[i] = n.output;
                    }

                    for (this.output_layer, 0..) |on, onum| {
                        for (0..NUM_HIDDEN) |i|
                            on.inputs[i] = hidden_outputs[i];
                        on.output = on.__tanh();

                        const err = case.outputs[onum] - on.output;
                        errors += std.math.pow(f32, err, 2);

                        on.error_gradient = on.output * (1.0 - on.output) * err;

                        for (this.hidden_layer, 0..) |n, i|
                            n.error_gradient = n.output * (1.0 - n.output) * on.error_gradient * on.weights[i];

                        for (this.hidden_layer) |n|
                            n.update_weights();
                        on.update_weights();
                    }
                }
                this.num_epochs += 1;
                // std.time.sleep(1e9 * 0.1);
                if (this.num_epochs % EPOCH_PRINT_THRESHOLD == 0)
                    std.debug.print("epoch {d}: sum squared errors: {d:.4}\n", .{ this.num_epochs, errors });

                if (errors > 3.0 or this.num_epochs >= EPOCH_THRESHOLD) {
                    this.num_epochs = 0;
                    this.seed = @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
                    rng = RndGen.init(this.seed);

                    for (this.hidden_layer) |n|
                        n.roll_weights(&rng);

                    for (this.output_layer) |n|
                        n.roll_weights(&rng);

                    start = std.time.milliTimestamp();
                } else if (errors < ERROR_THRESHOLD) {
                    this.runtime_ms = std.time.milliTimestamp() - start;
                    break;
                }
            }
        }

        fn test_model(this: *This) !void {
            for (this.test_cases, 0..) |case, case_num| {
                const hidden_outputs = try this.allocator.alloc(f32, NUM_HIDDEN);
                defer this.allocator.free(hidden_outputs);

                for (0..this.hidden_layer.len) |i| {
                    for (0..case.inputs.len) |j|
                        this.hidden_layer[i].inputs[j] = case.inputs[j];
                    this.hidden_layer[i].output = this.hidden_layer[i].__tanh();
                    hidden_outputs[i] = this.hidden_layer[i].output;
                }

                for (this.output_layer, 0..) |on, onum| {
                    for (0..NUM_HIDDEN) |i|
                        on.inputs[i] = hidden_outputs[i];

                    on.output = on.__tanh();
                    this.test_results[case_num][onum] = on.output * std.math.sign(on.output);
                }
            }
        }

        fn write_results_to_file(this: *This) !void {
            var file = try std.fs.cwd().createFile("results.txt", .{ .truncate = false });
            defer file.close();

            try file.seekFromEnd(0);
            const writer = file.writer();
            try writer.print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ RESULTS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n", .{});
            try writer.print("\tseed: {d}\n", .{this.seed});
            try writer.print("\truntime (ms): {d}ms\n", .{this.runtime_ms});
            try writer.print("\ttotal epochs: {d}\n", .{this.num_epochs});
            try writer.print("\ttotal hidden neurons: {d}\n", .{NUM_HIDDEN});
            try writer.print("\talpha: {d}\n", .{ALPHA});
            try writer.print("\tsum squared errors: {d}\n", .{ERROR_THRESHOLD});
            for (this.hidden_layer, 1..) |n, i| {
                try writer.print("\n\tn{d}| wb={d:.3}", .{ i, n.weight_bias });
                for (1..this.hidden_layer.len + 1, 0..) |j, k|
                    try writer.print("\tw{d}={d:.3} ", .{ j, n.weights[k] });
            }
            try writer.print("\n\n", .{});
            for (this.test_cases, 0..) |case, case_num| {
                try writer.print("\tcase: [", .{});
                for (0..NUM_INPUTS) |i|
                    if (i == NUM_INPUTS - 1) try writer.print("{d:.0}] ", .{case.inputs[i]}) else try writer.print("{d:.0}, ", .{case.inputs[i]});
                try writer.print("expected: [", .{});
                for (0..NUM_OUTPUTS) |i|
                    if (i == NUM_OUTPUTS - 1) try writer.print("{d:.0}] ", .{case.outputs[i]}) else try writer.print("{d:.0}, ", .{case.outputs[i]});

                try writer.print("actual: [", .{});
                for (0..NUM_OUTPUTS) |i|
                    if (i == NUM_OUTPUTS - 1) try writer.print("{d:.4}] ", .{this.test_results[case_num][i]}) else try writer.print("{d:.4}, ", .{this.test_results[case_num][i]});

                try writer.print("rounded: [", .{});
                for (0..NUM_OUTPUTS) |i|
                    if (i == NUM_OUTPUTS - 1) try writer.print("{d:.0}]\n", .{this.test_results[case_num][i]}) else try writer.print("{d:.0}, ", .{this.test_results[case_num][i]});
            }
            try writer.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
            try file.seekFromEnd(0);
        }

        fn print_results(this: *This) void {
            std.debug.print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ RESULTS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n", .{});
            std.debug.print("\tseed: {d}\n", .{this.seed});
            std.debug.print("\truntime (ms): {d}ms\n", .{this.runtime_ms});
            std.debug.print("\ttotal epochs: {d}\n", .{this.num_epochs});
            std.debug.print("\ttotal hidden neurons: {d}\n", .{NUM_HIDDEN});
            std.debug.print("\talpha: {d}\n", .{ALPHA});
            std.debug.print("\tsum squared errors: {d}\n", .{ERROR_THRESHOLD});
            for (this.hidden_layer, 1..) |n, i| {
                std.debug.print("\n\tn{d}| wb={d:.3}", .{ i, n.weight_bias });
                for (1..this.hidden_layer.len + 1, 0..) |j, k|
                    std.debug.print("\tw{d}={d:.3} ", .{ j, n.weights[k] });
            }
            std.debug.print("\n\n", .{});
            for (this.test_cases, 0..) |case, case_num| {
                std.debug.print("\tcase: [", .{});
                for (0..NUM_INPUTS) |i|
                    if (i == NUM_INPUTS - 1) std.debug.print("{d:.0}] ", .{case.inputs[i]}) else std.debug.print("{d:.0}, ", .{case.inputs[i]});
                std.debug.print("expected: [", .{});
                for (0..NUM_OUTPUTS) |i|
                    if (i == NUM_OUTPUTS - 1) std.debug.print("{d:.0}] ", .{case.outputs[i]}) else std.debug.print("{d:.0}, ", .{case.outputs[i]});
                std.debug.print("actual: [", .{});
                for (0..NUM_OUTPUTS) |i|
                    if (i == NUM_OUTPUTS - 1) std.debug.print("{d:.4}] ", .{this.test_results[case_num][i]}) else std.debug.print("{d:.4}, ", .{this.test_results[case_num][i]});
                std.debug.print("rounded: [", .{});
                for (0..NUM_OUTPUTS) |i|
                    if (i == NUM_OUTPUTS - 1) std.debug.print("{d:.0}]\n", .{this.test_results[case_num][i]}) else std.debug.print("{d:.0}, ", .{this.test_results[case_num][i]});
            }
            std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
        }
    };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const cases = try allocator.create([NUM_TEST_CASES]TestCase);
    cases.* = .{
        TestCase{ .inputs = .{ 1.0, 1.0 }, .outputs = .{0.0} },
        TestCase{ .inputs = .{ 1.0, 0.0 }, .outputs = .{1.0} },
        TestCase{ .inputs = .{ 0.0, 1.0 }, .outputs = .{1.0} },
        TestCase{ .inputs = .{ 0.0, 0.0 }, .outputs = .{0.0} },
    };

    var nn = try NeuralNetwork(f32).init(allocator, cases);
    defer nn.deinit();

    try nn.train();
    try nn.test_model();

    nn.print_results();
    // try nn.write_results_to_file(file);
}
