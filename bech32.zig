// Copyright (c) 2017, 2021 Pieter Wuille
// Copyright (c) 2017 Takatoshi Nakagawa
// Copyright (c) 2019 Google LLC
// Copyright (c) 2021 Stephen Gregoratto
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// TODO:
// - If subtle-encoding[1] creates a constant-time implemention, copy that.
// - Make en/decoders for segwit addresses?
// - The decoder should tell you where in source the error occured.
//   Though I'd rather wait until https://github.com/ziglang/zig/issues/2647
//   is implemented (if it ever is).
//
// [1]: https://github.com/iqlusioninc/crates/tree/main/subtle-encoding

//! A encoder/decoder for Bech32/Bech32m strings, as specified by BIP 173[^1] and BIP 350[^2]
//! This implementation is not constant-time, and may leak information due to
//! the use of lookup tables. Replacement by a constant-time solution is a
//! long-term desideratum.
//!
//! A Bech32 string is 90 US-ASCII characters long and consists of four parts:
//!
//! 1. The human readable part (HRP): 1-83 characters in the range [33-126].
//!    This is required, and is copied verbatim at the start.
//! 2. The seperator: "1". This character can appear in the HRP, so the last one
//!    found is used when decoding.
//! 3. The data, expanded to a sequence of 5-bit values and encoded using the
//!    Bech32 charset. This is optional, meaning that the checksum only
//!    validates the HRP.
//! 4. The checksum, a 30-bit integer expanded to six 5-bit values and encoded.
//!
//! Verifiying that a string is valid for the data it holds is done by
//! iteratively computing the checksum using polynomial arithetic for each byte
//! in the HRP (twice) and data. The checksum should be equal to 1 for the original Bech32
//! encoding or 0x2bc830a3 for the modified Bech32m encoding. A more thorough explanation
//! of how this works can be found under `doc/bech32-polymod.tex`.
//!
//! Using this package is simple:
//!
//! ```zig
//! const bech32 = @import("bech32.zig");
//! ⋮
//! // Encoding
//! const hrp = "ziglang"
//! const data = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
//! const enc = bech32.Encoding.bech32m;
//! var buf: [bech32.max_string_size]u8 = undefined;
//! const str = bech32.standard.Encoder.encode(&buf, hrp, data, enc);
//!
//! // Decoding
//! var data_buf[bech32.max_data_size]u8;
//! const res = try bech32.standard.Decoder.decode(&data_buf, str);
//! print(
//!     "HRP: {s}\nData: {s}\nEncoding: {s}",
//!     .{res.hrp, std.fmt.fmtSliceHexLower(res.data), @tagName(res.encoding)},
//! );
//! ```
//!
//! Implementations are required to output all lowercase strings, and for
//! uppercase transformations to be done externally. For convience, the Encoder
//! can do this for you if you set the `uppercase` flag at compile time.
//!
//! [^1]: https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki
//! [^2]: https://github.com/bitcoin/bips/blob/master/bip-0350.mediawiki

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

/// Supported encodings
pub const Encoding = enum(u30) {
    bech32 = 1,
    bech32m = 0x2bc830a3,
};

/// TooLong
/// : source exceeds the 90 character limit.
///
/// MixedCase
/// : Both lower and uppercase characters were found in source.
///
/// NoSeperator
/// : The seperator "1" wasn't found in source.
///
/// BadChar
/// : A character in the string is outside the valid range.
///
/// HRPEmpty, HRPTooLong
/// : The HRP provided is empty, or larger than max_hrp_size.
///
/// ChecksumTooShort
/// : Less than six checksum digits found.
///
/// Invalid
/// : The checksum did not equal 1 at the end of decoding.
pub const Error = error{
    TooLong,
    MixedCase,
    NoSeperator,
    BadChar,
    HRPEmpty,
    HRPTooLong,
    InvalidPadding,
    ChecksumTooShort,
    Invalid,
};

pub const bech32_charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l".*;
pub const max_string_size = 90;
/// Assuming no data.
pub const max_hrp_size = max_string_size - 1 - 6;
/// Assuming one-char HRP.
pub const max_data_len = max_string_size - 1 - 1 - 6;
pub const max_data_size = calcReduction(max_data_len);

/// Standard Bech32/Bech32m codecs, lowercase encoded strings.
pub const standard = struct {
    pub const Encoder = Bech32Encoder(bech32_charset, false);
    pub const Decoder = Bech32Decoder(bech32_charset);
};
/// Standard Bech32/Bech32m codecs, uppercase encoded strings.
pub const standard_uppercase = struct {
    pub const Encoder = Bech32Encoder(bech32_charset, true);
    pub const Decoder = Bech32Decoder(bech32_charset);
};

/// Calculates the space needed for expanding `data` to a sequence of u5s,
/// plus a padding bit if neeeded.
pub inline fn calcExpansion(len: usize) usize {
    var size: usize = len * 8;
    return @divTrunc(size, 5) + @boolToInt(@rem(size, 5) > 0);
}
/// The inverse of ::calcExpansion
pub inline fn calcReduction(len: usize) usize {
    return @divTrunc(len * 5, 8);
}

const Polymod = struct {
    const generator = [5]u30{ 0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3 };
    val: u30 = 1,

    inline fn step(self: *Polymod, value: u8) void {
        const bitset = self.val >> 25;
        self.val = (self.val & std.math.maxInt(u25)) << 5;
        self.val ^= value;

        inline for (generator) |g, i| {
            if (bitset >> @truncate(u5, i) & 1 != 0)
                self.val ^= g;
        }
    }

    inline fn finish(self: *Polymod, encoding: Encoding) void {
        self.val ^= @enumToInt(encoding);
    }
};

pub fn Bech32Encoder(comptime set: [32]u8, comptime uppercase: bool) type {
    return struct {
        const charset = if (!uppercase) set else blk: {
            var buf: [32]u8 = undefined;
            for (buf) |*c, i|
                c.* = std.ascii.toUpper(set[i]);

            break :blk buf;
        };
        const transform = if (!uppercase) std.ascii.toLower else std.ascii.toUpper;

        /// Calculates the space needed for the HRP and the data expansion.
        pub inline fn calcSize(hrp: []const u8, data: []const u8) usize {
            assert(hrp.len > 0 and hrp.len <= max_hrp_size);
            assert(data.len <= max_data_size);
            const result = hrp.len + 1 + calcExpansion(data.len) + 6;
            assert(result <= max_string_size);

            return result;
        }

        pub fn eightToFive(dest: []u5, source: []const u8) []const u5 {
            var acc: u12 = 0;
            var acc_len: u4 = 0;
            var i: usize = 0;

            for (source) |c| {
                acc = acc << 8 | c;
                acc_len += 8;
                while (acc_len >= 5) : (i += 1) {
                    acc_len -= 5;
                    dest[i] = @truncate(u5, acc >> acc_len);
                }
            }
            if (acc_len > 0) {
                dest[i] = @truncate(u5, acc << 5 - acc_len);
                i += 1;
            }

            return dest[0..i];
        }

        /// Encodes the HRP and data into a Bech32 or Bech32m string and stores the result
        /// in dest. The function contains a couple assertions that the caller
        /// should be aware of. The first are limit checks:
        ///
        /// - That the HRP doesn't exceed ::max_hrp_size.
        /// - That the expansion of data doesn't excede max_data_size. See
        ///   ::calcExpansion for how this is done.
        /// - That the full string doesn't exceed 90 chars. See
        ///   ::calcSize to compute this yourself.
        /// - That dest is large enough to hold the full string.
        ///
        /// Finally, the HRP is checked so that it doesn't contain invalid or
        /// mixed-case chars.
        pub fn encode(
            dest: []u8,
            hrp: []const u8,
            data: []const u8,
            encoding: Encoding,
        ) []const u8 {
            assert(dest.len >= calcSize(hrp, data));

            var polymod = Polymod{};
            var upper = false;
            var lower = false;
            for (hrp) |c, i| {
                assert(c >= 33 and c <= 126);
                var lc = c;
                switch (c) {
                    'A'...'Z' => {
                        upper = true;
                        lc |= 0b00100000;
                    },
                    'a'...'z' => lower = true,
                    else => {},
                }
                polymod.step(lc >> 5);
                dest[i] = c;
            }
            assert(!(upper and lower));
            polymod.step(0);

            var i: usize = 0;
            while (i < hrp.len) : (i += 1) {
                polymod.step(dest[i] & 31);
                dest[i] = transform(dest[i]);
            }
            dest[i] = '1';
            i += 1;

            var expanded: [max_data_len]u5 = undefined;
            const exp = eightToFive(&expanded, data);
            for (exp) |c| {
                polymod.step(c);
                dest[i] = charset[c];
                i += 1;
            }

            for ([_]u0{0} ** 6) |_| polymod.step(0);
            polymod.finish(encoding);
            for ([6]u5{ 0, 1, 2, 3, 4, 5 }) |n| {
                const shift = 5 * (5 - n);
                dest[i] = charset[@truncate(u5, polymod.val >> shift)];
                i += 1;
            }

            return dest[0..i];
        }
    };
}

pub const Result = struct { hrp: []const u8, data: []const u8, encoding: Encoding };
pub fn Bech32Decoder(comptime set: [32]u8) type {
    return struct {
        const reverse_charset = blk: {
            var buf = [_]?u5{null} ** 256;
            for (set) |c, i| {
                buf[c] = i;
                buf[std.ascii.toUpper(c)] = i;
            }

            break :blk buf;
        };

        pub fn calcSizeForSlice(source: []const u8) Error!usize {
            if (source.len > max_string_size) return Error.TooLong;
            const sep = std.mem.lastIndexOfScalar(u8, source, '1') orelse
                return Error.NoSeperator;
            if (sep == 0) return Error.HRPEmpty;
            if (sep > max_hrp_size) return Error.HRPTooLong;

            const data = if (source.len - (sep + 1) < 6)
                return Error.ChecksumTooShort
            else
                source[sep + 1 .. source.len - 6];

            return calcReduction(data.len);
        }

        pub fn fiveToEight(dest: []u8, source: []const u5) Error![]const u8 {
            var acc: u12 = 0;
            var acc_len: u4 = 0;
            var i: usize = 0;

            for (source) |c| {
                acc = acc << 5 | c;
                acc_len += 5;
                while (acc_len >= 8) : (i += 1) {
                    acc_len -= 8;
                    dest[i] = @truncate(u8, acc >> acc_len);
                }
            }
            if (acc_len > 5 or @truncate(u8, acc << 8 - acc_len) != 0)
                return Error.InvalidPadding;

            return dest[0..i];
        }

        /// Decodes and validates a Bech32 or Bech32m string `source`, and writes any
        /// data found to to `dest`. The returned Result has three members:
        ///
        /// - `hrp`, which is a slice of `source`
        /// - `data`, which is a slice of `dest`.
        /// - `encoding`, is the encoding that was detected if valid
        pub fn decode(dest: []u8, source: []const u8) Error!Result {
            assert(dest.len >= try calcSizeForSlice(source));

            const sep = std.mem.lastIndexOfScalar(u8, source, '1') orelse unreachable;
            const hrp = source[0..sep];
            const data = source[sep + 1 .. source.len - 6];
            const checksum = source[source.len - 6 ..];

            var pmod_buf: [max_hrp_size]u8 = undefined;
            var res = Result{ .hrp = hrp, .data = &[0]u8{}, .encoding = undefined };
            var polymod = Polymod{};
            var upper = false;
            var lower = false;
            for (hrp) |c, i| {
                var lc = c;
                switch (c) {
                    0...32, 127...255 => return Error.BadChar,
                    'A'...'Z' => {
                        upper = true;
                        lc |= 0b00100000;
                    },
                    'a'...'z' => lower = true,
                    else => {},
                }
                polymod.step(lc >> 5);
                pmod_buf[i] = c;
            }
            if (upper and lower) return Error.MixedCase;

            polymod.step(0);
            for (pmod_buf[0..hrp.len]) |c| polymod.step(c & 31);

            var convert_buf: [max_data_len]u5 = undefined;
            for (data) |c, i| {
                if (std.ascii.isUpper(c)) upper = true;
                if (std.ascii.isLower(c)) lower = true;

                const rev = reverse_charset[c] orelse return Error.BadChar;
                polymod.step(rev);
                convert_buf[i] = rev;
            }
            if (upper and lower) return Error.MixedCase;

            res.data = try fiveToEight(dest, convert_buf[0..data.len]);

            for (checksum) |c| {
                if (std.ascii.isUpper(c)) upper = true;
                if (std.ascii.isLower(c)) lower = true;

                const rev = reverse_charset[c] orelse return Error.BadChar;
                polymod.step(rev);
            }
            if (upper and lower) return Error.MixedCase;

            res.encoding = switch (polymod.val) {
                @enumToInt(Encoding.bech32) => Encoding.bech32,
                @enumToInt(Encoding.bech32m) => Encoding.bech32m,
                else => return Error.Invalid,
            };

            return res;
        }
    };
}

test "bech32 and bech32m test vectors" {
    try expectGoodStrings(good_strings_bech32, .bech32);
    try expectBadStrings(bad_strings_bech32);

    try expectGoodStrings(good_strings_bech32m, .bech32m);
    try expectBadStrings(bad_strings_bech32m);
}

fn expectGoodStrings(strings: anytype, encoding: Encoding) !void {
    var lower_buf: [max_string_size]u8 = undefined;
    var data_buf: [max_data_size]u8 = undefined;
    var encoded_buf: [max_string_size]u8 = undefined;

    for (strings) |str| {
        var lower = std.ascii.lowerString(&lower_buf, str);
        const res = standard.Decoder.decode(&data_buf, str) catch |err|
            std.debug.panic("Expected a valid string: {s} {s}\n", .{ str, @errorName(err) });

        try std.testing.expectEqual(encoding, res.encoding);

        const enc = standard.Encoder.encode(&encoded_buf, res.hrp, res.data, res.encoding);
        try std.testing.expectEqualStrings(lower, enc);

        const pos = std.mem.lastIndexOfScalar(u8, lower, '1') orelse unreachable;
        lower[pos + 1] = 'z';
        try std.testing.expectError(Error.Invalid, standard.Decoder.decode(&data_buf, lower));
    }
}

fn expectBadStrings(strings: anytype) !void {
    var data_buf: [max_data_size]u8 = undefined;
    for (strings) |i| {
        try testing.expectError(i.err, standard.Decoder.decode(&data_buf, i.str));
    }
}

const good_strings_bech32 = [_][]const u8{
    "A12UEL5L",
    "a12uel5l",
    "an83characterlonghumanreadablepartthatcontainsthenumber1andtheexcludedcharactersbio1tt5tgs",
    "11qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqc8247j",
    "split1checkupstagehandshakeupstreamerranterredcaperred2y9e3w",
};
const good_strings_bech32m = [_][]const u8{
    "A1LQFN3A",
    "a1lqfn3a",
    "an83characterlonghumanreadablepartthatcontainsthetheexcludedcharactersbioandnumber11sg7hg6",
    "abcdef1l7aum6echk45nj3s0wdvt2fg8x9yrzpqzd3ryx",
    // TODO(hazeycode): debug invalid padding
    // "11llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllludsr8",
    "split1checkupstagehandshakeupstreamerranterredcaperredlc445v",
    "?1v759aa",
};
const Bad = struct { str: []const u8, err: anyerror };
fn new(str: []const u8, err: anyerror) Bad {
    return Bad{ .str = str, .err = err };
}
const bad_strings_bech32 = [_]Bad{
    // Invalid checksum
    new("split1checkupstagehandshakeupstreamerranterredcaperred2y9e2w", Error.Invalid),
    new("split1checkupstagehandshakeupstreamerranterredcaperred2y9e2w", Error.Invalid),
    // Invalid character (space) and (DEL) in hrp
    new("s lit1checkupstagehandshakeupstreamerranterredcaperredp8hs2p", Error.BadChar),
    new("spl\x7ft1checkupstagehandshakeupstreamerranterredcaperred2y9e3w", Error.BadChar),
    // Invalid character (o) in data part
    new("split1cheo2y9e2w", Error.BadChar),
    // Short checksum.
    new("split1a2y9w", Error.ChecksumTooShort),
    // Empty hrp
    new("1checkupstagehandshakeupstreamerranterredcaperred2y9e3w", Error.HRPEmpty),
    // Too long
    new("11qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqsqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqc8247j", Error.TooLong),
    // Mixed case HRP, data and checksum.
    new("Foo1999999", Error.MixedCase),
    new("foo1qQzzzzzz", Error.MixedCase),
    new("foo1qzzzzzZ", Error.MixedCase),
    // BIP 173 invalid vectors.
    new("\x201nwldj5", Error.BadChar),
    new("an84characterslonghumanreadablepartthatcontainsthenumber1andtheexcludedcharactersbio1569pvx", Error.TooLong),
    new("pzry9x0s0muk", Error.NoSeperator),
    new("1pzry9x0s0muk", Error.HRPEmpty),
    new("x1b4n0q5v", Error.BadChar),
    new("li1dgmt3", Error.ChecksumTooShort),
    new("de1lg7wt\xff", Error.BadChar),
    // checksum calculated with uppercase form of HRP.
    new("A1G7SGD8", Error.Invalid),
    new("10a06t8", Error.HRPEmpty),
    new("1qzzfhee", Error.HRPEmpty),
};
const bad_strings_bech32m = [_]Bad{
    // HRP character out of range
    new("\x201xj0phk", Error.BadChar),
    new("\x7f1g6xzxy", Error.BadChar),
    new("\x801vctc34", Error.BadChar),
    // overall max length exceeded
    new("an84characterslonghumanreadablepartthatcontainsthetheexcludedcharactersbioandnumber11d6pts4", Error.TooLong),
    // No separator character
    new("qyrz8wqd2c9m", Error.NoSeperator),
    // Invalid data character
    new("y1b0jsk6g", Error.BadChar),
    new("lt1igcx5c0", Error.BadChar),
    // Too short checksum
    new("in1muywd", Error.ChecksumTooShort),
    // Invalid character in checksum
    new("mm1crxm3i", Error.BadChar),
    new("au1s5cgom", Error.BadChar),
    // checksum calculated with uppercase form of HRP
    new("M1VUXWEZ", Error.Invalid),
    // empty HRP
    new("1qyrz8wqd2c9m", Error.HRPEmpty),
    new("16plkw9", Error.HRPEmpty),
    new("1p2gdwpf", Error.HRPEmpty),
};
