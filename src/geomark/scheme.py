"""Watermark scheme parameters and the seeding function s_t = F(coarse history, [medium context], time bin).

Seeding (paper Sec. Method): C_t = run-collapse(par_{r-2}(c_1..c_{t-1})); h_t = last k elements of C_t;
s_t = (h_t, m_t, b) with m_t = par_{r-1}(c_{t-1}) if ctx == 'mid' else 0, and b = start-time bin.
exact_kgw baseline: s_t = ('x', c_{t-1}) (standard KGW with context width 1 on the exact previous fine cell).
exact_kgw_bin (Stage 7 factorial ablation): s_t = ('x', c_{t-1}, b) (exact previous fine cell plus the start-time bin).
Masking policies: M0 none; M1 skip steps whose (s_t, c_{t-1}) already occurred in the trace; M2 (Stage 7, seed-level):
skip the first step (empty history) and every step whose seed s_t already occurred earlier in the trace, so every
unmasked step has a seed that is fresh within the trace (exact per-trace distortion-freeness of Gumbel-H over the key).
eos_neutral (Stage 7): the sampler keeps the unwatermarked stop probability p(EOS) and watermarks only the choice of the
next cell conditional on continuing (KGW); for Gumbel-H EOS already forms its own medium group.
"""
from dataclasses import dataclass, asdict, replace
from .hexgrid import parent

METHODS = ('none', 'kgw', 'gumbel', 'exact_kgw', 'exact_kgw_bin', 'posthoc')


@dataclass(frozen=True)
class Scheme:
    method: str = 'kgw'        # none | kgw (GeoMark-KGW-2R) | gumbel (GeoMark-Gumbel-H) | exact_kgw | posthoc
    res: int = 9               # fine resolution r
    gamma: float = 0.25        # green fraction
    delta: float = 2.0         # fine-level bias
    delta_mid: float = 1.0     # medium-level (r-1) bias (KGW-2R); 0 disables the medium embedding
    k: int = 1                 # coarse history length (0 = time bin only)
    bin_min: int = 30          # time-bin width in minutes (0 = no time bin); multiple of 30
    ctx: str = 'coarse'        # 'coarse' (r-2 history only) | 'mid' (adds par_{r-1}(c_{t-1}))
    mask: str = 'M0'           # 'M0' no masking | 'M1' skip steps whose (s_t, c_{t-1}) already occurred in the trace
    two_res: bool = True       # detector: Fisher-combine fine and medium evidence
    coarse_off: int = 2        # coarse resolution = r - coarse_off
    eos_neutral: bool = False  # Stage 7: preserve p(EOS) exactly, watermark only the cell choice given continuation
    adj_neutral: bool = False  # Stage 8 (step-neutral): also preserve the total mass of non-adjacent moves; bias only the
                               # choice among H3 neighbours of the previous cell; the detector scores adjacent steps only

    @property
    def coarse_res(self):
        return self.res - self.coarse_off

    @property
    def mid_res(self):
        return self.res - 1

    @property
    def score(self):
        return 'gumbel' if self.method == 'gumbel' else 'kgw'

    @property
    def uses_mid(self):
        return self.two_res and self.method in ('kgw', 'gumbel', 'posthoc', 'exact_kgw_bin')   # exact_kgw_bin: Stage 8 (2R factorial cell)

    def bin_of_slot(self, slot):
        return (int(slot) * 30) // self.bin_min if self.bin_min else 0

    def to_dict(self):
        return asdict(self)

    def with_(self, **kw):
        return replace(self, **kw)

    @staticmethod
    def from_dict(d):
        d = {k: v for k, v in d.items() if k in Scheme.__dataclass_fields__}
        return Scheme(**d)


class SeedState:
    """Incremental seed computation along a fine-cell sequence (used by the sampler and the detector)."""
    __slots__ = ('sch', 'b', 'coarse', 'prev')

    def __init__(self, sch, slot):
        self.sch, self.b = sch, sch.bin_of_slot(slot)
        self.coarse, self.prev = [], 0

    def seed(self):
        s = self.sch
        if s.method == 'exact_kgw':
            return ('x', self.prev)
        if s.method == 'exact_kgw_bin':
            return ('x', self.prev, self.b)
        h = tuple(self.coarse[-s.k:]) if s.k > 0 else ()
        m = parent(self.prev, s.mid_res) if (s.ctx == 'mid' and self.prev) else 0
        return (h, m, self.b)

    def update(self, cell):
        cell = int(cell)
        if not self.sch.method.startswith('exact'):
            cp = parent(cell, self.sch.coarse_res)
            if not self.coarse or self.coarse[-1] != cp:
                self.coarse.append(cp)
                if len(self.coarse) > 8:
                    del self.coarse[0]
        self.prev = cell


def seeds_for_cells(cells, slot, sch):
    """Seed for every position of a collapsed fine-cell sequence, and the 'masked' flag of each position (M1)."""
    st = SeedState(sch, slot)
    seeds, masked, seen = [], [], set()
    for t, c in enumerate(cells):
        s = st.seed()
        masked.append(is_masked(sch, t, s, st.prev, seen))
        seeds.append(s)
        st.update(c)
    return seeds, masked


def is_masked(sch, t, s, prev, seen):
    """Masking decision at position t with seed s and previous cell prev; `seen` is updated in place.
    Shared by the sampler and the detector so both make identical decisions."""
    if sch.mask == 'M1':
        ctx = (s, prev)
        m = ctx in seen
        seen.add(ctx)
        return m
    if sch.mask == 'M2':
        m = t == 0 or s in seen
        seen.add(s)
        return m
    return False
