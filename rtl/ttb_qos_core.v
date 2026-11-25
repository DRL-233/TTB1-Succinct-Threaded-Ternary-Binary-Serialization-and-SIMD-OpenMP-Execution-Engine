`timescale 1ns / 1ps

// ============================================================================
// 1. 1-Cycle 5-Trit Succinct Wire Decoder with Zero-Byte Vacuum Skip
// ============================================================================
module ttb_trit_lut_decoder (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid_in,
    input  wire [7:0] packed_byte,
    output reg        valid_out,
    output reg        vacuum_skip,    // High when packed_byte == 0 (all 5 trits == 0)
    output reg        invalid_symbol, // High when packed_byte >= 243
    output reg  [1:0] trit0, trit1, trit2, trit3, trit4
);
    reg [9:0] lut_out;
    integer v;
    always @(*) begin
        if (packed_byte == 8'd0) begin
            lut_out = 10'd0;
        end else if (packed_byte < 8'd243) begin
            v = packed_byte;
            lut_out[1:0] = v % 3; v = v / 3;
            lut_out[3:2] = v % 3; v = v / 3;
            lut_out[5:4] = v % 3; v = v / 3;
            lut_out[7:6] = v % 3; v = v / 3;
            lut_out[9:8] = v % 3;
        end else begin
            lut_out = 10'b11_11_11_11_11;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out      <= 1'b0;
            vacuum_skip    <= 1'b0;
            invalid_symbol <= 1'b0;
            {trit4, trit3, trit2, trit1, trit0} <= 10'd0;
        end else begin
            valid_out      <= valid_in;
            vacuum_skip    <= valid_in && (packed_byte == 8'd0);
            invalid_symbol <= valid_in && (packed_byte >= 8'd243);
            {trit4, trit3, trit2, trit1, trit0} <= lut_out;
        end
    end
endmodule

// ============================================================================
// 2. Zero-DSP Dyadic Bounce Scaler: chi_0 * x = (15/8) * x = (x << 1) - (x >>> 3)
// ============================================================================
module chi0_dyadic_scaler #(
    parameter WIDTH = 32
)(
    input  wire signed [WIDTH-1:0] x_in,
    output wire signed [WIDTH-1:0] chi0_scaled_out
);
    assign chi0_scaled_out = (x_in <<< 1) - (x_in >>> 3);
endmodule

// ============================================================================
// 3. Zero-DSP Hardware OZIP psi1(s, g, i) = 9s + 3g + i Bijection (Fin 18)
// ============================================================================
module ozip_psi1_router (
    input  wire [1:0] s,
    input  wire [1:0] g,
    input  wire [1:0] i,
    output wire [4:0] psi1_idx
);
    assign psi1_idx = ({3'd0, s} << 3) + {3'd0, s} + ({3'd0, g} << 1) + {3'd0, g} + {3'd0, i};
endmodule

// ============================================================================
// 4. Event-Driven q=0.5 Instanton Clockwork Trigger & Power-Gating Master
// ============================================================================
module instanton_clockwork_trigger #(
    parameter [31:0] CHI0_Q16_THRESH = 32'h0001_E000, // 15/8 = 1.875 in Q16.16
    parameter [31:0] PREEMPT_Q16     = 32'h0001_C800  // 95% of 15/8 = 1.78125 in Q16.16
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sample_valid,
    input  wire [31:0] chi_amp_q16,
    output reg         tile_clk_en,        // Power-gating wake enable for OZIP systolic tile
    output reg         qec_preempt_pulse,  // Fires preemptive pi/7 stabilizer
    output reg         quantum_switch_out  // Zero-jitter clockwork switch output
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_clk_en        <= 1'b0;
            qec_preempt_pulse  <= 1'b0;
            quantum_switch_out <= 1'b0;
        end else begin
            tile_clk_en        <= sample_valid && (chi_amp_q16 >= PREEMPT_Q16);
            qec_preempt_pulse  <= sample_valid && (chi_amp_q16 >= PREEMPT_Q16);
            quantum_switch_out <= sample_valid && (chi_amp_q16 >= CHI0_Q16_THRESH);
        end
    end
endmodule

// ============================================================================
// 5. Clock-Gated 1-Cycle Parallel <3, 3, 2> Z3-Triality Contraction Tile
// ============================================================================
module ozip_triality_tile_3x3x2 #(
    parameter WIDTH = 32
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      clk_en, // Driven by instanton_clockwork_trigger or host burst
    input  wire                      start,
    input  wire signed [9*WIDTH-1:0] A_flat,
    input  wire signed [6*WIDTH-1:0] B_flat,
    output reg                       done,
    output reg  signed [6*WIDTH-1:0] C_flat
);
    wire signed [WIDTH-1:0] A [0:8];
    wire signed [WIDTH-1:0] B [0:5];

    genvar idx;
    generate
        for (idx = 0; idx < 9; idx = idx + 1) begin : unpack_A
            assign A[idx] = A_flat[idx*WIDTH +: WIDTH];
        end
        for (idx = 0; idx < 6; idx = idx + 1) begin : unpack_B
            assign B[idx] = B_flat[idx*WIDTH +: WIDTH];
        end
    endgenerate

    wire signed [WIDTH-1:0] c00 = A[0]*B[0] + A[1]*B[2] + A[2]*B[4];
    wire signed [WIDTH-1:0] c01 = A[0]*B[1] + A[1]*B[3] + A[2]*B[5];
    wire signed [WIDTH-1:0] c10 = A[3]*B[0] + A[4]*B[2] + A[5]*B[4];
    wire signed [WIDTH-1:0] c11 = A[3]*B[1] + A[4]*B[3] + A[5]*B[5];
    wire signed [WIDTH-1:0] c20 = A[6]*B[0] + A[7]*B[2] + A[8]*B[4];
    wire signed [WIDTH-1:0] c21 = A[6]*B[1] + A[7]*B[3] + A[8]*B[5];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done   <= 1'b0;
            C_flat <= {(6*WIDTH){1'b0}};
        end else if (clk_en) begin
            done <= start;
            if (start) begin
                C_flat[0*WIDTH +: WIDTH] <= c00;
                C_flat[1*WIDTH +: WIDTH] <= c01;
                C_flat[2*WIDTH +: WIDTH] <= c10;
                C_flat[3*WIDTH +: WIDTH] <= c11;
                C_flat[4*WIDTH +: WIDTH] <= c20;
                C_flat[5*WIDTH +: WIDTH] <= c21;
            end
        end else begin
            done <= 1'b0;
        end
    end
endmodule

// ============================================================================
// 6. Top-Level Unified TTB-QOS SoC Control Plane
// ============================================================================
module ttb_qos_top #(
    parameter WIDTH = 32
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      wire_valid,
    input  wire [7:0]                wire_byte,
    input  wire                      amp_valid,
    input  wire [31:0]               chi_amp_q16,
    input  wire                      force_tile_en,
    input  wire                      tile_start,
    input  wire signed [9*WIDTH-1:0] A_flat,
    input  wire signed [6*WIDTH-1:0] B_flat,
    output wire                      vacuum_skip,
    output wire                      qec_preempt_pulse,
    output wire                      quantum_switch_out,
    output wire                      tile_done,
    output wire signed [6*WIDTH-1:0] C_flat
);
    wire       lut_valid, invalid_sym;
    wire [1:0] t0, t1, t2, t3, t4;
    wire       trig_clk_en;

    ttb_trit_lut_decoder u_lut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(wire_valid), .packed_byte(wire_byte),
        .valid_out(lut_valid), .vacuum_skip(vacuum_skip),
        .invalid_symbol(invalid_sym),
        .trit0(t0), .trit1(t1), .trit2(t2), .trit3(t3), .trit4(t4)
    );

    instanton_clockwork_trigger u_trig (
        .clk(clk), .rst_n(rst_n),
        .sample_valid(amp_valid), .chi_amp_q16(chi_amp_q16),
        .tile_clk_en(trig_clk_en),
        .qec_preempt_pulse(qec_preempt_pulse),
        .quantum_switch_out(quantum_switch_out)
    );

    ozip_triality_tile_3x3x2 #(.WIDTH(WIDTH)) u_tile (
        .clk(clk), .rst_n(rst_n),
        .clk_en(trig_clk_en | force_tile_en),
        .start(tile_start), .A_flat(A_flat), .B_flat(B_flat),
        .done(tile_done), .C_flat(C_flat)
    );
endmodule
