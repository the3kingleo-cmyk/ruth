/*
 * ruth-core: Ruth's continuous latent brain as one dependency-free C99 file.
 *
 * Runs the exact step equations of brain/engine.py on a byte stream (the
 * text channel, dt = 1): field encoder -> sparse liquid core (CfC / LTC /
 * cortical columns) -> cerebellar forward model
 * -> selective SSM field -> Legendre sensory trace -> modern Hopfield
 * recall/storage (+ consolidation) -> RLS decoder + precision fusion.
 * O(1) work and memory per byte; weights live in CPU cache; no GPU.
 *
 *   make -C brain/csrc
 *   ./ruth export /tmp/ruth.bin
 *   brain/csrc/ruth-core /tmp/ruth.bin learn < notes.txt      (online learning, saves)
 *   printf 'I am ' | brain/csrc/ruth-core /tmp/ruth.bin gen 80
 *   brain/csrc/ruth-core /tmp/ruth.bin trace < file           (predictions, for parity)
 *
 * Sleep-time replay rehearsal lives in the Python engine only.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

/* ------------------------------------------------------------------ io */
typedef struct { char *name; uint32_t n; double *d; } Arr;
static Arr A[256];
static int NA;

static void die(const char *m, const char *x) { fprintf(stderr, "ruth-core: %s %s\n", m, x ? x : ""); exit(2); }

static void load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open", path);
    char magic[8]; uint32_t cnt;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "RUTHBRN1", 8)) die("bad model file", path);
    if (fread(&cnt, 4, 1, f) != 1 || cnt > 256) die("bad header", path);
    for (uint32_t i = 0; i < cnt; i++) {
        uint32_t ln, n;
        if (fread(&ln, 4, 1, f) != 1 || ln > 200) die("bad name", path);
        A[NA].name = calloc(ln + 1, 1);
        if (fread(A[NA].name, 1, ln, f) != ln || fread(&n, 4, 1, f) != 1) die("truncated", path);
        A[NA].n = n;
        A[NA].d = malloc(sizeof(double) * (n ? n : 1));
        if (fread(A[NA].d, sizeof(double), n, f) != n) die("truncated data", A[NA].name);
        NA++;
    }
    fclose(f);
}

static void save(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) die("cannot write", tmp);
    uint32_t cnt = (uint32_t)NA;
    fwrite("RUTHBRN1", 1, 8, f);
    fwrite(&cnt, 4, 1, f);
    for (int i = 0; i < NA; i++) {
        uint32_t ln = (uint32_t)strlen(A[i].name);
        fwrite(&ln, 4, 1, f); fwrite(A[i].name, 1, ln, f);
        fwrite(&A[i].n, 4, 1, f); fwrite(A[i].d, sizeof(double), A[i].n, f);
    }
    if (fclose(f) != 0) die("save failed", path);
#ifdef _WIN32
    remove(path);   /* Windows rename() will not replace an existing file */
#endif
    if (rename(tmp, path) != 0) die("save failed", path);
}

/* get array by name; n>=0 checks size; create=1 makes a zero array if absent */
static double *get(const char *name, long n, int create) {
    for (int i = 0; i < NA; i++)
        if (!strcmp(A[i].name, name)) {
            if (n >= 0 && (long)A[i].n != n) die("size mismatch for", name);
            return A[i].d;
        }
    if (!create) die("missing array", name);
    A[NA].name = strdup(name); A[NA].n = (uint32_t)n; A[NA].d = calloc(n ? n : 1, sizeof(double));
    return A[NA++].d;
}
static double S(const char *k) { char b[128]; snprintf(b, sizeof b, "cfg.%s", k); return get(b, 1, 0)[0]; }

/* ------------------------------------------------------------ model */
static int IN, OUT, TIN, TOUT, N, D, SS, L, KD, P, CELL, CB, UNF, F, H, Q, CQ, EQ, TL, TID, WC, LC, KEYLEN;
static double TW, CWT, EWT, FDEC, CBG, CBL, DTS, HOM, ATG, TMIN, TMAX, BETA, MERGE, LAM, SK, PRATE, *STEPS, *HASLIVE;
static int ITERS, WARM, ASLEEP, VOUT;
static double INHIBIT;
static double *omega, *w1, *b1, *w2;                 /* encoder */
static double *bf, *bg, *bh, *rev, *tau, *x, *vm, *act;   /* core */
static double *fa, *fs, *cbw, *cbp, *lfast, *lfevid, *lfeat;   /* fast weights, cerebellum */
static double *pin, *pdt, *bdt, *wb, *wc, *rate, *skip, *hs, *o; /* ssm */
static double *wk, *wv, *wm, *lk, *lv, *lm, *lcount, *luse, *mmeta; /* memory */
static double *W, *Pm, *pvar, *pstats, *ev, *kproj, *tidx, *tridx, *tad, *tbd, *tm, *cad, *cbd, *cm, *ead, *ebd, *em;
static double *lphi, *lpred, *lkey, *lwpred, *lrec, *levid, *epi;
static int NEPI;
/* sparse CSR copies of the three liquid heads */
typedef struct { int *rp, *ci; double *v; } CSR;
static CSR cf, cg, ch;

static CSR csr(const double *dense, int rows, int cols) {
    CSR m; int nz = 0;
    for (int i = 0; i < rows * cols; i++) nz += dense[i] != 0.0;
    m.rp = malloc(sizeof(int) * (rows + 1)); m.ci = malloc(sizeof(int) * (nz + 1));
    m.v = malloc(sizeof(double) * (nz + 1));
    nz = 0;
    for (int r = 0; r < rows; r++) {
        m.rp[r] = nz;
        for (int c = 0; c < cols; c++) {
            double w = dense[(long)r * cols + c];
            if (w != 0.0) { m.ci[nz] = c; m.v[nz++] = w; }
        }
    }
    m.rp[rows] = nz;
    return m;
}
static void spmv(const CSR *m, int rows, const double *v, const double *b, double *out) {
    for (int r = 0; r < rows; r++) {
        double s = 0.0;
        for (int k = m->rp[r]; k < m->rp[r + 1]; k++) s += m->v[k] * v[m->ci[k]];
        out[r] = s + b[r];
    }
}
static void mv(const double *M, int rows, int cols, const double *v, double *out) {
    for (int r = 0; r < rows; r++) {
        const double *row = M + (long)r * cols; double s = 0.0;
        for (int c = 0; c < cols; c++) s += row[c] * v[c];
        out[r] = s;
    }
}
static double sigm(double z) { return 0.5 * (1.0 + tanh(0.5 * z)); }
static double softplus(double z) { return (z > 0 ? z : 0) + log1p(exp(-fabs(z))); }
static void unit(double *v, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += v[i] * v[i];
    s = sqrt(s); if (s > 0) for (int i = 0; i < n; i++) v[i] /= s;
}

static void bind(void) {
    IN = (int)S("input_dim"); OUT = (int)S("output_dim"); TIN = (int)S("text_in"); TOUT = (int)S("text_out");
    N = (int)S("neurons"); D = (int)S("ssm_channels"); SS = (int)S("ssm_state"); L = (int)S("latent");
    KD = (int)S("key_dim"); P = (int)S("phi_dim"); CELL = (int)S("cell"); UNF = (int)S("ltc_unfolds");
    FDEC = S("fast_decay"); CB = (int)S("cerebellum"); CBG = S("cerebellar_gain"); CBL = S("cerebellum_lambda");
    F = (int)S("fourier"); H = (int)S("encoder_hidden"); Q = (int)S("lmu_order"); TL = (int)S("trace_lines");
    CQ = (int)S("context_order"); CWT = S("context_weight");
    EQ = (int)S("episode_order"); EWT = S("episode_weight");
    TID = (int)S("trace_in_decoder"); TW = S("thought_weight"); DTS = S("dt_scale"); HOM = S("homeostasis");
    ATG = S("activity_target"); TMIN = S("tau_min"); TMAX = S("tau_max"); BETA = S("beta");
    ITERS = (int)S("hopfield_iters"); MERGE = S("merge_threshold"); LAM = S("rls_lambda");
    SK = S("surprise_k"); WARM = (int)S("warmup"); ASLEEP = (int)S("auto_sleep"); PRATE = S("prec_rate");
    VOUT = (int)S("valence_out"); INHIBIT = S("inhibit");
    STEPS = get("cfg.steps", 1, 0); HASLIVE = get("cfg.has_live", 1, 0);
    int V = L + D + N, KEY = TL * Q + TL * CQ + TL * EQ + KD;
    KEYLEN = KEY;
    omega = get("enc.omega", (long)F * 9, 0); w1 = get("enc.w1", (long)H * (9 + 2 * F), 0);
    b1 = get("enc.b1", H, 0); w2 = get("enc.w2", (long)L * H, 0);
    cf = csr(get("core.wf", (long)N * V, 0), N, V); cg = csr(get("core.wg", (long)N * V, 0), N, V);
    ch = csr(get("core.wh", (long)N * V, 0), N, V);
    bf = get("core.bf", N, 0); bg = get("core.bg", N, 0); bh = get("core.bh", N, 0);
    rev = get("core.reversal", N, 0); tau = get("core.tau", N, 0); x = get("core.x", N, 0);
    act = get("core.activity", N, 0); vm = get("core.v", N, 0);
    fa = get("fast.a", (long)OUT * 2 * KEY, 0); fs = get("fast.s", (long)OUT * 2 * KEY, 0);
    cbw = get("cb.w", (long)N * (N + L + 1), 0); cbp = get("cb.p", (long)(N + L + 1) * (N + L + 1), 0);
    pin = get("ssm.p_in", (long)D * N, 0); pdt = get("ssm.p_dt", (long)D * N, 0); bdt = get("ssm.b_dt", D, 0);
    wb = get("ssm.w_b", (long)SS * N, 0); wc = get("ssm.w_c", (long)SS * N, 0);
    rate = get("ssm.rate", (long)D * SS, 0); skip = get("ssm.skip", D, 0);
    hs = get("ssm.h", (long)D * SS, 0); o = get("ssm.o", D, 0);
    mmeta = get("mem.meta", 3, 0);
    for (int i = 0; i < NA; i++) {
        if (!strcmp(A[i].name, "mem.wk")) WC = (int)(A[i].n / KEY);
        if (!strcmp(A[i].name, "mem.lk")) LC = (int)(A[i].n / KEY);
    }
    wk = get("mem.wk", (long)WC * KEY, 0); wv = get("mem.wv", (long)WC * OUT, 0);
    wm = get("mem.wm", (long)WC * OUT, 0);
    lk = get("mem.lk", (long)LC * KEY, 0); lv = get("mem.lv", (long)LC * OUT, 0);
    lm = get("mem.lm", (long)LC * OUT, 0);
    lcount = get("mem.l_count", LC, 0); luse = get("mem.l_use", LC, 0);
    W = get("rls.w", (long)OUT * P, 0); Pm = get("rls.p", (long)P * P, 0);
    pvar = get("prec.var", OUT, 0); pstats = get("prec.stats", 2, 0); ev = get("expert_var", 3L * OUT, 0);
    kproj = get("key_proj", (long)KD * (N + D), 0); tidx = get("target_index", OUT, 0);
    tridx = get("trace.index", TL, 0); tad = get("trace.ad", (long)Q * Q, 0);
    tbd = get("trace.bd", Q, 0); tm = get("trace.m", (long)Q * TL, 0);
    cad = get("ctx.ad", (long)CQ * CQ, 0); cbd = get("ctx.bd", CQ, 0); cm = get("ctx.m", (long)CQ * TL, 0);
    if (EQ) { ead = get("ep3.ad", (long)EQ * EQ, 0); ebd = get("ep3.bd", EQ, 0); em = get("ep3.m", (long)EQ * TL, 0); }
    lphi = get("live.phi", P, 1); lpred = get("live.pred", OUT, 1); lkey = get("live.key", KEY, 1);
    lwpred = get("live.wpred", OUT, 1); lrec = get("live.recall", OUT, 1);
    levid = get("live.evidence", OUT, 1);
    lfast = get("live.fast", OUT, 1); lfevid = get("live.fevid", OUT, 1); lfeat = get("live.feat", 2L * KEY, 1);
    epi = get("episodic_index", -1, 0);
    for (NEPI = 0; epi[NEPI] >= 0; NEPI++) {}
}

/* ------------------------------------------------------------ memory */
static void store(const double *key, const double *val, const double *mask) {
    int KEY = KEYLEN, i = (int)mmeta[1];
    memcpy(wk + (long)i * KEY, key, sizeof(double) * KEY); unit(wk + (long)i * KEY, KEY);
    memcpy(wv + (long)i * OUT, val, sizeof(double) * OUT);
    memcpy(wm + (long)i * OUT, mask, sizeof(double) * OUT);
    mmeta[1] = (i + 1) % WC;
    mmeta[0] = mmeta[0] + 1 < WC ? mmeta[0] + 1 : WC;
}

static void consolidate(void) {
    int KEY = KEYLEN, wn = (int)mmeta[0], ln = (int)mmeta[2];
    for (int i = 0; i < wn; i++) {
        double *key = wk + (long)i * KEY, *val = wv + (long)i * OUT, *obs = wm + (long)i * OUT;
        int j = -1;
        if (ln) {
            double best = -INFINITY; int bj = 0;
            for (int r = 0; r < ln; r++) {
                double s = 0; for (int c = 0; c < KEY; c++) s += lk[(long)r * KEY + c] * key[c];
                if (s > best) { best = s; bj = r; }
            }
            if (best >= MERGE) {
                double cnt = lcount[bj];
                for (int c = 0; c < KEY; c++) lk[(long)bj * KEY + c] = (cnt * lk[(long)bj * KEY + c] + key[c]) / (cnt + 1);
                unit(lk + (long)bj * KEY, KEY);
                for (int c = 0; c < OUT; c++) {
                    double e = lm[(long)bj * OUT + c], tot = e + obs[c];
                    if (tot > 0) lv[(long)bj * OUT + c] = (e * lv[(long)bj * OUT + c] + obs[c] * val[c]) / (tot > 1e-12 ? tot : 1e-12);
                    lm[(long)bj * OUT + c] = tot;
                }
                lcount[bj] = cnt + 1;
                continue;
            }
        }
        if (ln < LC) j = ln++;
        else {
            double lo = INFINITY;
            for (int r = 0; r < ln; r++) if (luse[r] + lcount[r] < lo) { lo = luse[r] + lcount[r]; j = r; }
        }
        memcpy(lk + (long)j * KEY, key, sizeof(double) * KEY);
        memcpy(lv + (long)j * OUT, val, sizeof(double) * OUT);
        memcpy(lm + (long)j * OUT, obs, sizeof(double) * OUT);
        lcount[j] = 1.0; luse[j] = 0.0;
    }
    mmeta[0] = mmeta[1] = 0; mmeta[2] = ln;
    for (int r = 0; r < LC; r++) luse[r] *= 0.5;
}

static double recall(const double *q, int track, double *out_r, double *out_w) {
    int KEY = KEYLEN, wn = (int)mmeta[0], ln = (int)mmeta[2], n = wn + ln;
    memset(out_r, 0, sizeof(double) * OUT);
    memset(out_w, 0, sizeof(double) * OUT);
    if (n == 0) return 0.0;
    static double *xi, *xn, *sc; static int cap;
    if (cap < n || !xi) {
        free(sc); sc = malloc(sizeof(double) * (WC + LC));
        free(xi); xi = malloc(sizeof(double) * KEY); free(xn); xn = malloc(sizeof(double) * KEY); cap = WC + LC;
    }
    memcpy(xi, q, sizeof(double) * KEY); unit(xi, KEY);
    int it = ITERS > 1 ? ITERS : 1;
    for (int t = 0; t < it; t++) {
        double mx = -INFINITY, sum = 0;
        for (int r = 0; r < n; r++) {
            const double *k = r < wn ? wk + (long)r * KEY : lk + (long)(r - wn) * KEY;
            double s = 0; for (int c = 0; c < KEY; c++) s += k[c] * xi[c];
            s = BETA * s + (r < wn ? 0.0 : log(lcount[r - wn]));
            sc[r] = s; if (s > mx) mx = s;
        }
        for (int r = 0; r < n; r++) { sc[r] = exp(sc[r] - mx); sum += sc[r]; }
        memset(xn, 0, sizeof(double) * KEY);
        for (int r = 0; r < n; r++) {
            sc[r] /= sum;
            const double *k = r < wn ? wk + (long)r * KEY : lk + (long)(r - wn) * KEY;
            for (int c = 0; c < KEY; c++) xn[c] += sc[r] * k[c];
        }
        memcpy(xi, xn, sizeof(double) * KEY);
    }
    double pmax = 0;
    for (int r = 0; r < n; r++) {
        const double *v = r < wn ? wv + (long)r * OUT : lv + (long)(r - wn) * OUT;
        for (int c = 0; c < OUT; c++) {
            double m = r < wn ? wm[(long)r * OUT + c] : lm[(long)(r - wn) * OUT + c];
            if (m > 1.0) m = 1.0;
            out_r[c] += sc[r] * m * v[c];
            out_w[c] += sc[r] * m;
        }
        if (sc[r] > pmax) pmax = sc[r];
        if (track && r >= wn) luse[r - wn] += sc[r];
    }
    for (int c = 0; c < OUT; c++) out_r[c] /= (out_w[c] > 1e-12 ? out_w[c] : 1e-12);
    return pmax;
}

/* ------------------------------------------------------------ step */
static void byte_code(int b, double *u) {
    for (int i = 0; i < 8; i++) u[i] = (b >> i) & 1 ? 1.0 : -1.0;
    u[8] = b / 127.5 - 1.0;
}
static int byte_decode(const double *v) { int b = 0; for (int i = 0; i < 8; i++) if (v[i] > 0) b |= 1 << i; return b; }

/* recursive least squares (a recursive pseudo-inverse): Wm (no x ni), Pmat (ni x ni) */
static void rls_update(double *Wm, double *Pmat, int ni, int no, const double *in, const double *err, double lam) {
    static double *pp, *k; static int cap;
    if (cap < ni) { free(pp); free(k); pp = malloc(sizeof(double) * ni); k = malloc(sizeof(double) * ni); cap = ni; }
    mv(Pmat, ni, ni, in, pp);
    double den = lam; for (int i = 0; i < ni; i++) den += in[i] * pp[i];
    for (int i = 0; i < ni; i++) k[i] = pp[i] / den;
    for (int r = 0; r < no; r++) for (int c = 0; c < ni; c++) Wm[(long)r * ni + c] += err[r] * k[c];
    for (int r = 0; r < ni; r++) {
        double kr = k[r], *row = Pmat + (long)r * ni;
        for (int c = 0; c < ni; c++) row[c] = (row[c] - kr * pp[c]) / lam;
    }
}

static void step(const double *u, int learn) {
    const double dt = 1.0;
    static double *msk, *tgt, *err, *werr, *evd, *psi, *xh, *xp, *ff, *fw, *fm, *vin, *fv, *gv, *hv, *y, *dl, *bb, *cc, *z, *eh, *ef, *th, *xo, *key, *r, *phi, *wp;
    int V = L + D + N, TD = TL * Q, CTD = TL * CQ, ETD = TL * EQ, KEY = KEYLEN;
    if (!msk) {
        psi = malloc(sizeof(double) * (N + L + 1)); xh = malloc(sizeof(double) * N); xp = malloc(sizeof(double) * N);
        ff = malloc(sizeof(double) * 2 * KEY); fw = malloc(sizeof(double) * OUT); fm = malloc(sizeof(double) * OUT);
        msk = malloc(sizeof(double) * OUT); evd = malloc(sizeof(double) * OUT); tgt = malloc(sizeof(double) * OUT); err = malloc(sizeof(double) * OUT); werr = malloc(sizeof(double) * OUT);
        vin = malloc(sizeof(double) * V); fv = malloc(sizeof(double) * N); gv = malloc(sizeof(double) * N);
        hv = malloc(sizeof(double) * N); y = malloc(sizeof(double) * D); dl = malloc(sizeof(double) * D);
        bb = malloc(sizeof(double) * SS); cc = malloc(sizeof(double) * SS); z = malloc(sizeof(double) * L);
        eh = malloc(sizeof(double) * H); ef = malloc(sizeof(double) * (9 + 2 * F)); th = malloc(sizeof(double) * KD);
        xo = malloc(sizeof(double) * (N + D)); key = malloc(sizeof(double) * KEY); r = malloc(sizeof(double) * OUT);
        phi = malloc(sizeof(double) * P); wp = malloc(sizeof(double) * OUT);
    }
    /* only the text lines are observed here; every other channel is silent
       and carries no evidence (masked), exactly as in the Python engine */
    double nobs = 0;
    for (int i = 0; i < OUT; i++) {
        tgt[i] = u[(int)tidx[i]];
        msk[i] = (i >= TOUT && i < TOUT + 9) ? 1.0 : 0.0;
        nobs += msk[i];
    }
    if (*HASLIVE) {
        for (int i = 0; i < OUT; i++) { err[i] = msk[i] * (tgt[i] - lpred[i]); werr[i] = msk[i] * (tgt[i] - lwpred[i]); }
        int outlier = 0;
        if (learn) {
            for (int i = 0; i < OUT; i++) {
                double r = PRATE * msk[i];
                ev[i] += r * (werr[i] * werr[i] - ev[i]); if (ev[i] < 1e-6) ev[i] = 1e-6;
                double d = tgt[i] - lrec[i];
                ev[OUT + i] += r * levid[i] * (d * d - ev[OUT + i]); if (ev[OUT + i] < 1e-6) ev[OUT + i] = 1e-6;
                double df = tgt[i] - lfast[i];
                ev[2 * OUT + i] += r * lfevid[i] * (df * df - ev[2 * OUT + i]); if (ev[2 * OUT + i] < 1e-6) ev[2 * OUT + i] = 1e-6;
            }
            double s = 0; for (int i = 0; i < OUT; i++) s += msk[i] * err[i] * err[i] / pvar[i];
            s = 0.5 * s / (nobs > 1 ? nobs : 1);
            outlier = s > pstats[0] + SK * sqrt(pstats[1] > 0 ? pstats[1] : 0);
            for (int i = 0; i < OUT; i++) { pvar[i] += PRATE * msk[i] * (err[i] * err[i] - pvar[i]); if (pvar[i] < 1e-6) pvar[i] = 1e-6; }
            double d = s - pstats[0]; pstats[0] += PRATE * d; pstats[1] += PRATE * (d * d - pstats[1]);
            rls_update(W, Pm, P, OUT, lphi, werr, LAM);          /* neocortex */
            {   /* hippocampal fast weights: dual-trace Hebbian */
                int FD = 2 * KEY;
                for (int r = 0; r < OUT; r++) {
                    double t = msk[r] * tgt[r], m = msk[r];
                    double *ar = fa + (long)r * FD, *sr = fs + (long)r * FD;
                    for (int c = 0; c < FD; c++) {
                        ar[c] = FDEC * ar[c] + (1.0 - FDEC) * t * lfeat[c];
                        sr[c] = FDEC * sr[c] + (1.0 - FDEC) * m * lfeat[c];
                    }
                }
            }
            if (outlier && *STEPS >= WARM) store(lkey, tgt, msk);
            if (ASLEEP && (int)mmeta[0] == WC) consolidate();
        }
    }
    /* field encoder on the text channel */
    const double *ut = u + TIN;
    for (int i = 0; i < 9; i++) ef[i] = ut[i];
    for (int f = 0; f < F; f++) {
        double s = 0; for (int i = 0; i < 9; i++) s += omega[f * 9 + i] * ut[i];
        ef[9 + f] = sin(s); ef[9 + F + f] = cos(s);
    }
    mv(w1, H, 9 + 2 * F, ef, eh); for (int i = 0; i < H; i++) eh[i] = tanh(eh[i] + b1[i]);
    mv(w2, L, H, eh, z); for (int i = 0; i < L; i++) z[i] = tanh(z[i]);
    /* cortex */
    memcpy(xp, x, sizeof(double) * N);
    memcpy(vin, z, sizeof(double) * L); memcpy(vin + L, o, sizeof(double) * D);
    if (CELL == 2) {   /* cortical columns: tau dv/dt = -v + W tanh(v) + W_in u + b */
        for (int i = 0; i < N; i++) vin[L + D + i] = tanh(vm[i]);
        spmv(&cf, N, vin, bf, fv);
        for (int i = 0; i < N; i++) {
            double a = exp(-dt / tau[i]);
            vm[i] = a * vm[i] + (1.0 - a) * fv[i];
            x[i] = tanh(vm[i]);
        }
    } else if (CELL == 1) {
        double h = dt / UNF;
        for (int t = 0; t < UNF; t++) {
            memcpy(vin + L + D, x, sizeof(double) * N);
            spmv(&cf, N, vin, bf, fv);
            for (int i = 0; i < N; i++) { double f = sigm(fv[i]); x[i] = (x[i] + h * f * rev[i]) / (1.0 + h * (1.0 / tau[i] + f)); }
        }
    } else {
        memcpy(vin + L + D, x, sizeof(double) * N);
        spmv(&cf, N, vin, bf, fv); spmv(&cg, N, vin, bg, gv); spmv(&ch, N, vin, bh, hv);
        for (int i = 0; i < N; i++) {
            double gate = sigm(-fv[i] * (dt / tau[i]));
            x[i] = gate * tanh(gv[i]) + (1.0 - gate) * tanh(hv[i]);
        }
    }
    if (HOM > 0) for (int i = 0; i < N; i++) {
        act[i] += 0.01 * (fabs(x[i]) - act[i]);
        tau[i] *= exp(HOM * (act[i] - ATG));
        if (tau[i] < TMIN) tau[i] = TMIN; if (tau[i] > TMAX) tau[i] = TMAX;
    }
    /* cerebellum: forward model from last state + efference copy */
    if (CB) {
        int ni = N + L + 1;
        memcpy(psi, xp, sizeof(double) * N); memcpy(psi + N, z, sizeof(double) * L); psi[N + L] = 1.0;
        mv(cbw, N, ni, psi, xh);
        for (int i = 0; i < N; i++) xh[i] -= x[i];              /* e = anticipated - actual */
        if (learn) {
            for (int i = 0; i < N; i++) xh[i] = -xh[i];
            rls_update(cbw, cbp, ni, N, psi, xh, CBL);
        } else if (CBG > 0) {
            for (int i = 0; i < N; i++) { double v = x[i] + CBG * xh[i]; x[i] = v > 1 ? 1 : (v < -1 ? -1 : v); }
        }
    }
    /* selective SSM field */
    mv(pin, D, N, x, y); mv(pdt, D, N, x, dl); mv(wb, SS, N, x, bb); mv(wc, SS, N, x, cc);
    for (int s = 0; s < SS; s++) bb[s] = tanh(bb[s]);
    double rs = sqrt((double)SS);
    for (int d = 0; d < D; d++) {
        double delta = softplus(dl[d] + bdt[d]) * (dt * DTS), acc = 0;
        for (int s = 0; s < SS; s++) {
            double dec = exp(-rate[d * SS + s] * delta);
            double *hp = hs + d * SS + s;
            *hp = dec * *hp + (1.0 - dec) * (y[d] * bb[s]);
            acc += *hp * cc[s];
        }
        o[d] = tanh(acc / rs + skip[d] * y[d]);
    }
    /* Legendre traces (fast + context): m = Ad m + outer(Bd, u[trace]) */
    {
        int mq = Q > CQ ? Q : CQ; if (EQ > mq) mq = EQ;
        static double *nm; if (!nm) nm = malloc(sizeof(double) * mq * TL);
        for (int i = 0; i < Q; i++) for (int c = 0; c < TL; c++) {
            double s = 0; for (int k = 0; k < Q; k++) s += tad[i * Q + k] * tm[k * TL + c];
            nm[i * TL + c] = s + tbd[i] * u[(int)tridx[c]];
        }
        memcpy(tm, nm, sizeof(double) * Q * TL);
        for (int i = 0; i < CQ; i++) for (int c = 0; c < TL; c++) {
            double s = 0; for (int k = 0; k < CQ; k++) s += cad[i * CQ + k] * cm[k * TL + c];
            nm[i * TL + c] = s + cbd[i] * u[(int)tridx[c]];
        }
        memcpy(cm, nm, sizeof(double) * CQ * TL);
        for (int i = 0; i < EQ; i++) for (int c = 0; c < TL; c++) {
            double s = 0; for (int k = 0; k < EQ; k++) s += ead[i * EQ + k] * em[k * TL + c];
            nm[i * TL + c] = s + ebd[i] * u[(int)tridx[c]];
        }
        if (EQ) memcpy(em, nm, sizeof(double) * EQ * TL);
    }
    /* key = [unit(trace), tw * unit(key_proj [x, o])] */
    memcpy(xo, x, sizeof(double) * N); memcpy(xo + N, o, sizeof(double) * D);
    mv(kproj, KD, N + D, xo, th); unit(th, KD);
    memcpy(key, tm, sizeof(double) * TD); unit(key, TD);
    memcpy(key + TD, cm, sizeof(double) * CTD); unit(key + TD, CTD);
    for (int i = 0; i < CTD; i++) key[TD + i] *= CWT;
    if (EQ) {
        memcpy(key + TD + CTD, em, sizeof(double) * ETD); unit(key + TD + CTD, ETD);
        for (int i = 0; i < ETD; i++) key[TD + CTD + i] *= EWT;
    }
    for (int i = 0; i < KD; i++) key[TD + CTD + ETD + i] = TW * th[i];
    double conf = recall(key, learn, r, evd);
    /* fast-weight recall of the recent past */
    {
        int FD = 2 * KEY; double nrm = 0;
        for (int i = 0; i < KEY; i++) {
            double a = key[i] > 0 ? key[i] : 0, b = key[i] < 0 ? -key[i] : 0;
            ff[i] = a * a; ff[KEY + i] = b * b;
        }
        for (int c = 0; c < FD; c++) nrm += ff[c] * ff[c];
        if (nrm == 0) nrm = 1.0;
        for (int r2 = 0; r2 < OUT; r2++) {
            double num = 0, mass = 0;
            const double *ar = fa + (long)r2 * FD, *sr = fs + (long)r2 * FD;
            for (int c = 0; c < FD; c++) { num += ar[c] * ff[c]; mass += sr[c] * ff[c]; }
            fw[r2] = num / (mass > 1e-12 ? mass : 1e-12);
            double e = mass / nrm; fm[r2] = e < 0 ? 0 : (e > 1 ? 1 : e);
        }
    }
    /* phi and decoder */
    int p = 0;
    memcpy(phi + p, u, sizeof(double) * IN); p += IN;
    memcpy(phi + p, x, sizeof(double) * N); p += N;
    memcpy(phi + p, o, sizeof(double) * D); p += D;
    if (TID) { memcpy(phi + p, tm, sizeof(double) * TD); p += TD; }
    memcpy(phi + p, r, sizeof(double) * OUT); p += OUT;
    phi[p++] = conf; phi[p++] = 1.0;
    if (p != P) die("phi layout mismatch", NULL);
    mv(W, OUT, P, phi, wp);
    for (int i = 0; i < OUT; i++) {   /* precision-weighted fusion of three experts */
        double pw = 1.0 / ev[i], pr = conf > 0.0 ? evd[i] / ev[OUT + i] : 0.0, pf = fm[i] / ev[2 * OUT + i];
        lpred[i] = (pw * wp[i] + pr * r[i] + pf * fw[i]) / (pw + pr + pf);
    }
    for (int e = 0; e < NEPI; e++) { int i = (int)epi[e]; lpred[i] = evd[i] * r[i]; }
    memcpy(lphi, phi, sizeof(double) * P); memcpy(lkey, key, sizeof(double) * KEY);
    memcpy(lwpred, wp, sizeof(double) * OUT); memcpy(lrec, r, sizeof(double) * OUT);
    memcpy(levid, evd, sizeof(double) * OUT);
    memcpy(lfast, fw, sizeof(double) * OUT); memcpy(lfevid, fm, sizeof(double) * OUT);
    memcpy(lfeat, ff, sizeof(double) * 2 * KEYLEN);
    *HASLIVE = 1; *STEPS += 1;
}

static void feed(int b, int learn) {
    static double *u; if (!u) u = calloc(IN, sizeof(double));
    memset(u, 0, sizeof(double) * IN);
    byte_code(b, u + TIN);
    step(u, learn);
}

/* ------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ruth-core MODEL learn|trace|gen N [--no-learn] [--no-save] < input\n");
        return 2;
    }
    const char *model = argv[1], *cmd = argv[2];
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);    /* bytes are signals: no CRLF translation */
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    int learn = 1, dosave = 1;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--no-learn")) learn = 0;
        if (!strcmp(argv[i], "--no-save")) dosave = 0;
    }
    load(model); bind();
    long n = 0, ok = 0; int c;
    clock_t t0 = clock();
    if (!strcmp(cmd, "trace")) {
        while ((c = getchar()) != EOF) {
            feed(c, learn);
            for (int i = 0; i < 9; i++) printf(i ? " %.17g" : "%.17g", lpred[TOUT + i]);
            printf("\n");
        }
        return 0;
    }
    if (!strcmp(cmd, "learn")) {
        while ((c = getchar()) != EOF) {
            if (*HASLIVE) ok += byte_decode(lpred + TOUT) == c;
            feed(c, learn); n++;
        }
        double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
        fprintf(stderr, "learned %ld bytes, next-byte accuracy %.4f, %.0f B/s, long-term basins %d\n",
                n, n ? (double)ok / n : 0.0, n / (el > 0 ? el : 1e-9), (int)mmeta[2]);
        if (dosave && learn) save(model);
        return 0;
    }
    if (!strcmp(cmd, "gen")) {
        long want = argc > 3 ? atol(argv[3]) : 80;
        while ((c = getchar()) != EOF) feed(c, learn);
        if (!*HASLIVE) feed(' ', 0);
        int withheld = 0;
        for (long i = 0; i < want; i++) {
            /* discretion: she will not voice what she was told is private */
            if (VOUT >= 0 && lpred[VOUT] < -INHIBIT) { withheld = 1; break; }
            int b = byte_decode(lpred + TOUT);
            putchar(b);
            feed(b, 0);
        }
        putchar('\n');
        if (withheld) fprintf(stderr, "[withheld: private]\n");
        if (dosave && learn) save(model);
        return 0;
    }
    die("unknown command", cmd);
    return 2;
}
