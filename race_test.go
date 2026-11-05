// Copyright (c) 2026 the go-gfx/gpu2d authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build race

package gpu2d

// raceEnabled: the race detector allocates on its own, so allocation counts
// are only asserted without it.
const raceEnabled = true
