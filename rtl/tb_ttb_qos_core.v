`timescale 1ns / 1ps

module tb_ttb_qos_core;
    reg clk = 0;
    reg rst_n = 0;
    always #2.5 clk = ~clk; // 200 MHz (5.0 ns)

    reg                wire_valid = 0;
    reg  [7:0]         wire_byte  = 0;
    reg                amp_valid  = 0;
    reg  [31:0]        chi_amp_q16 = 0;
    reg                force_tile_en = 0;
    reg                tile_start = 0;
    reg  signed [287:0] A_flat;
    reg  signed [191:0] B_flat;

    wire               vacuum_skip;
    wire               qec_preempt_pulse;
    wire               quantum_switch_out;
    wire               tile_done;
    wire signed [191:0] C_flat;

    // Test Zero-DSP Dyadic Scaler: (15/8) * 1.0 (0x0001_0000) == 1.875 (0x0001_E000)
    wire signed [31:0] dyadic_out;
    chi0_dyadic_scaler #(.WIDTH(32)) u_dyadic (.x_in(32'sh0001_0000), .chi0_scaled_out(dyadic_out));

    ttb_qos_top #(.WIDTH(32)) u_top (
        .clk(clk), .rst_n(rst_n),
        .wire_valid(wire_valid), .wire_byte(wire_byte),
        .amp_valid(amp_valid), .chi_amp_q16(chi_amp_q16),
        .force_tile_en(force_tile_en), .tile_start(tile_start),
        .A_flat(A_flat), .B_flat(B_flat),
        .vacuum_skip(vacuum_skip),
        .qec_preempt_pulse(qec_preempt_pulse),
        .quantum_switch_out(quantum_switch_out),
        .tile_done(tile_done), .C_flat(C_flat)
    );

    wire signed [31:0] C00 = C_flat[ 31:  0];
    wire signed [31:0] C01 = C_flat[ 63: 32];
    wire signed [31:0] C10 = C_flat[ 95: 64];
    wire signed [31:0] C11 = C_flat[127: 96];
    wire signed [31:0] C20 = C_flat[159:128];
    wire signed [31:0] C21 = C_flat[191:160];

    initial begin
        #10 rst_n = 1;

        // 1. Verify Zero-DSP Dyadic Bounce Scaler: (15/8)*1.0 == 0x0001_E000
        if (dyadic_out !== 32'sh0001_E000) begin
            $display("FAIL: Dyadic scaler got 0x%08X", dyadic_out);
            $finish;
        end
        $display("PASS [Zero-DSP Dyadic Scaler] (15/8) * 1.0 = 0x%08X (0 DSPs)", dyadic_out);

        // 2. Verify Zero-Byte Vacuum Skip on wire_byte == 0
        @(posedge clk);
        wire_valid <= 1'b1;
        wire_byte  <= 8'd0;
        @(posedge clk);
        wire_valid <= 1'b0;
        #1;
        if (vacuum_skip !== 1'b1) begin
            $display("FAIL: Expected vacuum_skip=1 on byte 0");
            $finish;
        end
        $display("PASS [Zero-Byte Vacuum Skip]  byte 0 -> vacuum_skip=1 in 5.0 ns");

        // 3. Verify Event-Driven Self-Clocking Wake + 1-Cycle OZIP Tile Burst at chi_0 = 15/8
        A_flat = {32'sd0, 32'sd5, 32'sd1, -32'sd2, 32'sd4, 32'sd0, 32'sd3, -32'sd1, 32'sd2};
        B_flat = {-32'sd1, 32'sd1, 32'sd4, -32'sd2, 32'sd1, 32'sd3};

        // Pre-assert bounce threshold chi_0 = 1.875 (0x0001_E000) to wake tile_clk_en
        @(posedge clk);
        amp_valid   <= 1'b1;
        chi_amp_q16 <= 32'h0001_E000;
        @(posedge clk); // Cycle 1: trigger asserts tile_clk_en, preempt, switch
        tile_start  <= 1'b1;
        @(posedge clk); // Cycle 2: clock-gated OZIP tile executes burst
        amp_valid   <= 1'b0;
        tile_start  <= 1'b0;
        #1;
        if (qec_preempt_pulse !== 1'b1 || quantum_switch_out !== 1'b1 ||
            tile_done !== 1'b1 || C00 !== 32'sd11 || C21 !== 32'sd21) begin
            $display("FAIL: Event-driven SoC burst mismatch");
            $finish;
        end
        $display("PASS [Event-Driven SoC Burst] preempt=1, switch=1, C=[%0d, %0d, %0d, %0d, %0d, %0d] (200 MHz)",
                 C00, C01, C10, C11, C20, C21);
        $finish;
    end
endmodule
