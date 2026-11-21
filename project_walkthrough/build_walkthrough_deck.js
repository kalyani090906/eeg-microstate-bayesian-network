// build_walkthrough_deck.js - the project walkthrough deck (for the team's understanding, not the presentation).
// Every number here was read from MLproject/outputs (CSV files) or the code on 21 Nov 2026.
// Run (from MLproject\project_walkthrough):  npm install pptxgenjs@3.12.0   then   node build_walkthrough_deck.js

// ---------------- settings (change only here) ----------------
const path = require("path");
const HERE = __dirname;
// the .pptx is NOT kept in git: it is written outside the repo, to SEM5/ML/reports/project_walkthrough/
const OUT_DIR = path.join(HERE, "..", "..", "reports", "project_walkthrough");
const OUT = path.join(OUT_DIR, "EEG_Microstates_Project_Walkthrough_v2_21Nov2026.pptx");
const P = path.join(HERE, "..", "outputs");   // project figures
const F = path.join(HERE, "figures");                              // walkthrough figures
const INK = "222222", MUTED = "666666", ACCENT = "0F6E6E", TINT = "EEF5F5", LINE = "C9D6D6";
const FONT = "Calibri";
// --------------------------------------------------------------

const pptxgen = require("pptxgenjs");
require("fs").mkdirSync(OUT_DIR, { recursive: true });
const sizeOf = require("image-size");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 inches
pres.title = "EEG microstates - project walkthrough";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontFace: FONT, fontSize: 28, bold: true, color: INK, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, align: "right" },
});

let currentModule = "";
function base(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: currentModule });
  s.addText(title, { placeholder: "title" });
  s.addText(currentModule, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: FONT, fontSize: 10,
    color: MUTED, margin: 0, isTextBox: true });
  if (notes) s.addNotes(notes);
  return s;
}

// text with **bold** parts -> pptxgenjs runs
function runs(str, opts) {
  return str.split("**").map((t, i) => ({ text: t, options: { ...opts, bold: i % 2 === 1 } }))
    .filter(r => r.text.length);
}

// bullets: array of strings; "- " at the start = second level
function bullets(s, items, box, size = 16) {
  const out = [];
  items.forEach((raw, idx) => {
    const sub = raw.startsWith("- ");
    const txt = sub ? raw.slice(2) : raw;
    const r = runs(txt, { fontFace: FONT, fontSize: sub ? size - 2 : size, color: sub ? "444444" : INK });
    r[0].options = { ...r[0].options, bullet: sub ? { indent: 18 } : { indent: 18 },
      indentLevel: sub ? 1 : 0, paraSpaceAfter: sub ? 3 : 7 };
    if (idx < items.length - 1) r[r.length - 1].options.breakLine = true;
    out.push(...r);
  });
  s.addText(out, { ...box, valign: "top", margin: 2, isTextBox: true });
}

function fit(file, x, y, w, h) {
  const d = sizeOf(file);
  const r = Math.min(w / d.width, h / d.height);
  const iw = d.width * r, ih = d.height * r;
  return { path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}

function qaBox(s, q, a) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.0, w: 12.1, h: 0.95, fill: { color: TINT },
    line: { color: TINT }, rectRadius: 0.08 });
  s.addText([
    { text: "Likely question: ", options: { bold: true, color: ACCENT } },
    { text: q, options: { color: INK, breakLine: true } },
    { text: "Answer: ", options: { bold: true, color: ACCENT } },
    { text: a, options: { color: INK } },
  ], { x: 0.8, y: 6.03, w: 11.7, h: 0.9, fontFace: FONT, fontSize: 13, valign: "middle", margin: 0, isTextBox: true });
}

const fs = require("fs");
function readCsv(file) {
  // small CSV reader (handles quoted fields with commas)
  const parse = l => { const out = []; let cur = "", q = false;
    for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; }
    out.push(cur); return out; };
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const head = parse(lines[0]);
  return lines.slice(1).map(l => { const v = parse(l), o = {}; head.forEach((h, i) => o[h] = v[i]); return o; });
}

function peakTables() {
  const rows = readCsv(path.join(HERE, "numbers_per_subject.csv"));
  [["h", "controls"], ["s", "patients"]].forEach(([g, word]) => {
    const part = rows.filter(r => r.subject.startsWith(g));
    table(`Every person's numbers (${word})`, [
      ["Person", "Recording (min)", "Samples used (first 15 min)", "GFP peaks (whole, filtered)", "Peaks per s (filtered)", "ms between peaks", "Peaks per s (unfiltered)", "After 5% cut", "Used for the maps"],
      ...part.map(r => [r.subject, Number(r.minutes_whole).toFixed(1), Number(r.gfp_values_first_15min).toLocaleString("en-US"),
        Number(r.peaks_whole_filtered).toLocaleString("en-US"), Number(r.peaks_per_sec_filtered).toFixed(1),
        Number(r.ms_between_peaks_filtered).toFixed(0), Number(r.peaks_per_sec_unfiltered).toFixed(1),
        Number(r.peaks_after_dropping_top5pct).toLocaleString("en-US"), r.peaks_used_for_maps]),
    ], { colW: [0.9, 1.3, 1.6, 1.6, 1.3, 1.2, 1.4, 1.4, 1.4], rowH: 0.3, fs: 12,
      note: "From numbers_per_subject.csv (make_pipeline_figures.py). Recordings shorter than 15 min use what they have (e.g. s06: 12.3 min). Everybody gives exactly 1,500 peaks to the maps, whatever the length: s14 has 48,365 peaks from 36 min and still gives 1,500.", noteY: 6.3 });
  });
}

function foldTables(run) {
  const rows = readCsv(path.join(HERE, `fold_by_fold_${run.replace(" ", "_")}.csv`));
  const others = rows.length - 1;
  [["HC", "a control"], ["SC", "a patient"]].forEach(([g, who]) => {
    const part = rows.filter(r => r.true === g);
    table(`Fold by fold (${run}): the ${part.length} folds where ${who} is tested`, [
      ["Test person", "Training set", "Train acc.", "Features L1 kept", "P(patient) of test person", "Predicted", "Correct?"],
      ...part.map(r => [`${r.test_subject} (${r.true})`, `${others} others: ${r.train_HC} HC + ${r.train_SC} SC`,
        (100 * Number(r.train_accuracy)).toFixed(1) + "%", r.features_kept, Number(r.P_patient).toFixed(3),
        r.predicted, r.correct === "True" ? "yes" : "NO"]),
    ], { colW: [1.6, 2.6, 1.2, 2.6, 1.9, 1.1, 1.1], rowH: 0.3, fs: 12,
      note: "Main model: L1 selection + logistic regression, maps refitted in each fold. P(patient) >= 0.5 -> predicted SC. Train acc. = the same fold's model scored on its own training people. File: project_walkthrough/fold_by_fold_" + run.replace(" ", "_") + ".csv",
      noteY: 6.3 });
  });
}

// slide: bullets left, picture right (or picture below when wide)
function tf(title, items, img, opt = {}) {
  const s = base(title, opt.notes);
  const bottom = opt.qa ? 5.85 : 6.85;
  if (!img) {
    bullets(s, items, { x: 0.6, y: 1.4, w: 12.1, h: bottom - 1.4 }, opt.size || 17);
  } else if (opt.stack) {
    const th = opt.textH || 1.6;
    bullets(s, items, { x: 0.6, y: 1.35, w: 12.1, h: th }, opt.size || 15);
    s.addImage(fit(img, 0.6, 1.4 + th, 12.1, bottom - 1.45 - th - (opt.caption ? 0.3 : 0)));
  } else {
    const tw = opt.textW || 5.4;
    bullets(s, items, { x: 0.6, y: 1.4, w: tw, h: bottom - 1.4 }, opt.size || 16);
    s.addImage(fit(img, 0.6 + tw + 0.3, 1.4, 12.1 - tw - 0.3, bottom - 1.4 - (opt.caption ? 0.35 : 0)));
  }
  if (opt.caption) s.addText(opt.caption, { x: 0.6, y: bottom - 0.32, w: 12.1, h: 0.3, fontFace: FONT,
    fontSize: 11, italic: true, color: MUTED, align: "right", margin: 0, isTextBox: true });
  if (opt.qa) qaBox(s, opt.qa[0], opt.qa[1]);
  return s;
}

// slide: table (first row = header)
function table(title, rows, opt = {}) {
  const s = base(title, opt.notes);
  const fs = opt.fs || 13;
  const data = rows.map((r, i) => r.map(c => ({ text: String(c), options: {
    bold: i === 0, color: i === 0 ? "FFFFFF" : INK, fill: { color: i === 0 ? ACCENT : (i % 2 ? "FFFFFF" : "F5F8F8") },
    fontFace: FONT, fontSize: fs, valign: "middle" } })));
  const top = opt.intro ? 1.95 : 1.4;
  if (opt.intro) s.addText(runs(opt.intro, { fontFace: FONT, fontSize: 15, color: INK }),
    { x: 0.6, y: 1.3, w: 12.1, h: 0.55, margin: 0, valign: "middle", isTextBox: true });
  s.addTable(data, { x: 0.6, y: top, w: opt.w || 12.1, colW: opt.colW, border: { type: "solid", pt: 0.5, color: LINE },
    rowH: opt.rowH || 0.32, margin: 0.05 });
  if (opt.note) s.addText(runs(opt.note, { fontFace: FONT, fontSize: 13, color: "444444" }),
    { x: 0.6, y: opt.noteY || 5.3, w: 12.1, h: 0.6, margin: 0, valign: "top", isTextBox: true });
  if (opt.qa) qaBox(s, opt.qa[0], opt.qa[1]);
  return s;
}

// slide: flow of boxes with arrows; steps = [[title, sub], ...]; perRow boxes per row
function flow(title, steps, opt = {}) {
  const s = base(title, opt.notes);
  const perRow = opt.perRow || 4;
  const rowsN = Math.ceil(steps.length / perRow);
  const gap = 0.45, bw = (12.1 - gap * (perRow - 1)) / perRow;
  const bh = opt.bh || 1.35, top = opt.top || 1.55, rgap = opt.rgap || 0.55;
  steps.forEach((st, i) => {
    const r = Math.floor(i / perRow), c = i % perRow;
    const x = 0.6 + c * (bw + gap), y = top + r * (bh + rgap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h: bh, fill: { color: st[2] ? "FFF4E5" : TINT },
      line: { color: st[2] ? "E0A040" : LINE, width: 1 }, rectRadius: 0.1 });
    s.addText([{ text: st[0], options: { bold: true, fontSize: 15, color: ACCENT, breakLine: true } },
      { text: st[1], options: { fontSize: 13, color: INK } }],
      { x: x + 0.1, y: y + 0.05, w: bw - 0.2, h: bh - 0.1, fontFace: FONT, valign: "middle", align: "center", margin: 0, isTextBox: true });
    if (c < perRow - 1 && i < steps.length - 1)
      s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.07, y: y + bh / 2 - 0.13, w: gap - 0.14, h: 0.26,
        fill: { color: "9AB5B5" }, line: { color: "9AB5B5" } });
    if (c === perRow - 1 && i < steps.length - 1)
      s.addShape(pres.shapes.DOWN_ARROW, { x: x + bw / 2 - 0.13, y: y + bh + 0.06, w: 0.26, h: rgap - 0.12,
        fill: { color: "9AB5B5" }, line: { color: "9AB5B5" } });
  });
  if (opt.after) opt.after(s, top + rowsN * (bh + rgap));
  if (opt.note) s.addText(runs(opt.note, { fontFace: FONT, fontSize: 14, color: "444444" }),
    { x: 0.6, y: top + rowsN * (bh + rgap) - 0.2, w: 12.1, h: 0.8, margin: 0, valign: "top", isTextBox: true });
  if (opt.qa) qaBox(s, opt.qa[0], opt.qa[1]);
  return s;
}

// section divider: module number, title, what goes in / out, code files
let MODULE_NO = 0;
function section(_unused, title, goesIn, comesOut, files, notes) {
  const num = MODULE_NO++;
  currentModule = `Module ${num} - ${title}`;
  pres.addSection({ title: currentModule });
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: currentModule });
  s.addText(`MODULE ${num}`, { x: 0.6, y: 2.4, w: 6, h: 0.4, fontFace: FONT, fontSize: 16, bold: true,
    color: ACCENT, charSpacing: 3, margin: 0, isTextBox: true });
  s.addText(title, { x: 0.6, y: 2.85, w: 12.1, h: 1.0, fontFace: FONT, fontSize: 40, bold: true, color: INK, margin: 0, isTextBox: true });
  // one plain line with the code files, only when there are real files
  const real = files.filter(f => !/no code/i.test(f));
  if (real.length) s.addText("Code: " + real.join("   ·   "), { x: 0.6, y: 4.05, w: 12.1, h: 0.8, fontFace: FONT,
    fontSize: 15, color: MUTED, valign: "top", margin: 0, isTextBox: true });
  if (notes) s.addNotes(notes);
  s.addText(currentModule, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
}

/* =====================================================================================
   TITLE + MODULE 0
   ===================================================================================== */
{
  currentModule = "Start";
  pres.addSection({ title: "Start" });
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Start" });
  s.addText("EEG Microstates for Schizophrenia", { x: 0.6, y: 2.0, w: 12.1, h: 1.0, fontFace: FONT, fontSize: 44, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText("Project walkthrough: every step, every file, from the raw EEG to the results", { x: 0.6, y: 3.0, w: 12.1, h: 0.6, fontFace: FONT, fontSize: 22, color: ACCENT, margin: 0, isTextBox: true });
  s.addText("For the team's own understanding (not the presentation deck)  ·  23CSE301  ·  21 November 2026", { x: 0.6, y: 3.8, w: 12.1, h: 0.4, fontFace: FONT, fontSize: 15, color: MUTED, margin: 0, isTextBox: true });
  s.addText("Working title: Microstate Transition Dynamics for Schizophrenia Classification from Resting-State EEG", { x: 0.6, y: 5.6, w: 12.1, h: 0.4, fontFace: FONT, fontSize: 14, italic: true, color: MUTED, margin: 0, isTextBox: true });
}

section(0, "The big picture", ["28 EEG recordings (EDF files)", "14 patients, 14 healthy controls"],
  ["A prediction per person: patient or control", "A p-value: is that better than chance?", "Group statistics per feature"],
  ["All of MLproject/scripts and labs/", "This module: no code, only the map of the project"]);

flow("The whole pipeline in one picture", [
  ["1. Read EEG", "EDF file, 19 channels, 250 Hz"],
  ["2. Clean", "fix names, repair bad channels, 2-20 Hz, average reference"],
  ["3. Epochs", "5-s pieces, first 15 min, reject noisy ones"],
  ["4. GFP peaks", "moments where the scalp map is strongest"],
  ["5. 4 maps", "modified k-means on 42,000 peak maps -> A, B, C, D"],
  ["6. Label", "every sample gets its closest map -> A A B B D ..."],
  ["7. Features", "duration, occurrence, transitions: 24 -> 16 per person"],
  ["8. Use them", "rank-sum test (groups), classifiers (LOSO), permutation test", true],
], { perRow: 4, bh: 1.55, top: 1.6, rgap: 0.7,
  note: "Steps 1-7 never look at the diagnosis. Only step 8 uses it. Since November 2026, steps 5-7 are **redone inside every cross-validation fold** for the classifiers (Module 11).",
  notes: "In plain words: we record the brain's electrical activity, clean it, find the moments when the pattern on the scalp is clearest, group those patterns into 4 types, then describe each person by how long each type lasts, how often it appears and which type follows which. Then we ask: do patients and controls differ, and can a computer tell them apart?" });

table("Words you must be able to explain", [
  ["Word", "Meaning in this project"],
  ["EEG", "voltage on the scalp, recorded by electrodes; here 19 electrodes"],
  ["channel / electrode", "one recording point, named by the 10-20 system (Fp1, Cz, O2 ...)"],
  ["sample, sampling rate", "one measurement of all 19 channels; 250 Hz = 250 samples per second = one every 4 ms"],
  ["epoch", "a 5-second piece of the recording = 1,250 samples"],
  ["scalp map (topomap)", "the 19 values at one instant, drawn on a head picture"],
  ["microstate", "a scalp map that stays almost the same for about 60-120 ms, then switches"],
  ["GFP", "global field power: how strong the map is at one instant (spread of the 19 values)"],
  ["polarity", "the sign of the map; a map and its sign-flipped copy count as the same microstate"],
  ["GEV", "global explained variance: how much of the signal the 4 maps explain (ours 67.4%)"],
  ["LOSO", "leave-one-subject-out: train on 27 people, test on the 28th, 28 times"],
], { colW: [3.0, 9.1], rowH: 0.4, fs: 14 });

table("Which file does which step", [
  ["Step", "Module files (imported, never run)", "Scripts you run", "Writes"],
  ["Clean + epochs", "eeg_preprocessing.py", "check_bad_channels.py", "printed table"],
  ["GFP, maps, labels", "microstate_model.py", "tune_microstates_v3.py, gev_by_k.py, check_map_stability.py, name_states.py", "shared_microstate_model_v3.pkl, gev_by_k.csv"],
  ["Features", "microstate_features.py", "extract_epoch_features.py, make_final_features.py, check_features_match.py", "eeg_features_epoch_mean_15min.csv, features_final_16.csv"],
  ["Raw epochs", "eeg_preprocessing.py", "make_epoch_array.py", "epochs_15min.npz (420 MB, not in git)"],
  ["Group statistics", "-", "ranksum_group_test.py, group_specific_analysis.py, make_kim_style_tables.py", "outputs/group_specific/"],
  ["Scoring", "evaluate.py", "labs/lab09_a3_classifiers.py, item4_..., item7_...", "outputs/items/"],
  ["Leakage fix", "evaluate.py", "nested_loso_refit_maps.py, permutation_test_nested.py", "outputs/nested_loso/"],
  ["Deep learning", "-", "item2_embeddings.py, item3_finetune.py, item8_lstm_sequences.py", "outputs/items/, outputs/lstm/"],
  ["Toy demo", "-", "notebooks/make_toy_gfp_notebook.py", "notebooks/toy_gfp_demo.ipynb"],
], { colW: [2.0, 2.8, 4.4, 2.9], rowH: 0.45, fs: 12,
  note: "Every script: settings block at the top, a comment on each step, seed 42, finds its own files. Run from MLproject:  eeg_env\\Scripts\\python.exe scripts\\<name>.py", noteY: 6.3 });

table("The three core modules and their functions", [
  ["Module (scripts/)", "What it holds", "Functions"],
  ["eeg_preprocessing.py", "settings (bad channels, 2-20 Hz, 5 s, 15 min, z = 3.5) and the cleaning steps", "recording_files, subject_and_label, load_raw, filter_and_reference, load_clean, cut_epochs, kept_epochs"],
  ["microstate_model.py", "GFP, peak sampling, modified k-means fit, save/load, labelling with smoothing, A-D names", "global_field_power, sample_gfp_peaks, fit_shared_model, save_model, load_model, suggest_state_names, label_epoch, STATE_NAMES"],
  ["microstate_features.py", "the feature formulas, written out line by line", "segments, state_parameters, transition_counts, transition_probabilities, epoch_features, subject_features, feature_names"],
], { colW: [2.8, 4.3, 5.0], rowH: 0.75, fs: 13,
  note: "Each step and formula is defined **once**, and every pipeline script imports these modules. microstate_features.py gives the same numbers as the pycrostates library: scripts/check_features_match.py compares them on 345 real epochs (h01 and s01); the largest difference is 6e-17, i.e. identical.", noteY: 4.6 });

tf("How to set up and run the project", [
  "Windows, PowerShell, Python 3.13. From the MLproject folder:",
  "- python -m venv eeg_env",
  "- eeg_env\\Scripts\\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu",
  "Put the 28 EDF files in dataset/ (not in git; download from RepOD, DOI 10.18150/repod.0107441)",
  "Run any script as: **eeg_env\\Scripts\\python.exe scripts\\<name>.py** (labs: labs\\<name>.py)",
  "Every script has its settings at the top, uses seed 42, and finds its own files",
  "Large files not in git (rebuild them with the script named): shared_microstate_model_v3.pkl (tune_microstates_v3.py), epochs_15min.npz and epochs_15min_labram.npz (make_epoch_array.py), embeddings_labram.npz (item2_embeddings.py)",
  "No GPU needed: everything runs on a laptop CPU",
], null);

table("Run order 1: from the raw EEG to the feature tables", [
  ["#", "Script", "What it does", "Time"],
  ["1", "check_bad_channels.py", "lists flat / noisy / uncorrelated channels (prints only)", "minutes"],
  ["2", "tune_microstates_v3.py", "fits the one shared 4-map model (no labels)", "minutes"],
  ["3", "name_states.py, gev_by_k.py", "A-D names by shape; GEV for k = 2-8", "seconds; 10-20 min"],
  ["4", "check_map_stability.py", "refits with other seeds / subjects left out (prints only)", "about 10 min"],
  ["5", "extract_epoch_features.py", "epochs, rejection, labels, 24 features per epoch and per person", "about 15 min"],
  ["6", "make_final_features.py", "24 -> 16 features, with and without s14", "seconds"],
  ["7", "plot_labelled_maps.py, plot_transition_matrix.py", "figures of the maps and transitions", "seconds"],
  ["8", "make_epoch_array.py (and --preset labram)", "saves the kept raw epochs for later steps", "minutes"],
], { colW: [0.5, 4.2, 5.6, 1.8], rowH: 0.48, fs: 13,
  note: "Part A can be skipped: its outputs (the feature tables) are already in git.", noteY: 6.1 });

table("Run order 2: evaluation, items and the November additions", [
  ["#", "Script", "What it does", "Time"],
  ["10-11", "evaluate.py, labs\\lab09_a3_classifiers.py, baseline_24_vs_16.py", "scoring function; item 1 (11 classifiers); 24 vs 16", "seconds - 5 min"],
  ["12-13", "item2_embeddings.py, item3_finetune.py", "LaBraM frozen (item 2) and fine-tuned (item 3)", "2 min; about 1 hour"],
  ["14-16", "item4_feature_reduction.py, lab09_a4 / a5, item7_clustering.py", "items 4, 5 and 7", "about 1 min each"],
  ["17-19", "permutation_test.py, item6_comparison.py, collect_items.py", "chance check (old maps); item 6; copies results", "about 25 min"],
  ["20-23", "group_specific_analysis.py, ranksum_group_test.py, make_kim_style_tables.py, plot_group_maps_labelled.py", "group maps and group statistics", "20-30 min"],
  ["24", "nested_loso_refit_maps.py", "maps refitted in every fold, 12 classifiers (main results)", "about 1 hour"],
  ["25", "permutation_test_nested.py", "1,000 shuffles on the refitted results", "about 1.5 hours"],
  ["26", "item8_lstm_sequences.py", "LSTM on the sequences", "about 1 hour"],
  ["27", "notebooks\\make_toy_gfp_notebook.py", "writes and runs the toy GFP notebook", "1 min"],
], { colW: [0.8, 4.9, 4.4, 2.0], rowH: 0.48, fs: 12 });

/* =====================================================================================
   THEORY
   ===================================================================================== */
section(0, "EEG and microstate theory", ["what the electrodes measure"], ["what a microstate is (and is not)", "the 4 classes and their literature links", "what is known in schizophrenia"],
  ["no code: background for every later module"]);

tf("What EEG measures", [
  "The brain has billions of neurons that communicate with electrical and chemical signals",
  "When many neurons are active **together**, they create tiny voltage differences that reach the scalp; the electrodes pick these up (microvolts, µV)",
  "Voltage is always a **difference** between two points, so it is measured relative to a reference (we use the average reference, Module 4)",
  "**250 Hz** = 250 measurements per second on every electrode = one every **4 ms**",
  "At one instant the EEG is **19 numbers**. Drawn on a head, they form a **scalp map** (topography):",
  "- red = positive, blue = negative, white = close to zero",
  "- only the 19 electrode values are real; the colour between them is interpolated",
], path.join(F, "maps_with_names.png"), { textW: 6.0 });

tf("What a microstate is", [
  "Lehmann et al. (1987): the scalp map does **not** change smoothly",
  "- it stays almost the same (**quasi-stable**) for about **60-120 ms**",
  "- then it **switches abruptly** to another shape, stays, switches again",
  "Only a handful of shapes keep coming back, in **every** person: called the 'atoms of thought'",
  "**Why 'state'?** For a while the whole scalp keeps one stable spatial pattern",
  "**Why 'micro'?** It lasts only milliseconds (ordinary brain states last seconds or minutes)",
  "So a recording becomes a sequence: **A A A C C C C B B D D D A ...**",
  "A person is then described by how that sequence behaves: how long each state lasts, how often it appears, how much time it fills, which state follows which",
], path.join(F, "backfitting_epoch.png"), { textW: 5.6 });

tf("What microstates are NOT", [
  "**Not brain regions.** A microstate is a pattern on the scalp, not one physical part of the brain. Links to brain networks are associations",
  "**Not frequency bands.** Bands (delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30 Hz) say **how fast** the signal oscillates; microstates say **what spatial pattern** is on the scalp",
  "**Not single electrodes.** Fp1 alone is not a microstate; a microstate uses all 19 electrodes together",
  "**Why the millisecond timing matters**: a power spectrum summarises minutes of signal; microstates show which pattern is present, for how long, and which comes next",
], null, { qa: ["Is microstate C 'in the front of the brain'?", "No. It is a scalp pattern with a front-back orientation. Many different sources can produce it."] });

tf("Polarity: a map and its flipped copy are one state", [
  "The EEG oscillates: half a cycle later the same source gives the same map with **all signs flipped**",
  "Example: Fp1 +10 -> -10, Fp2 +5 -> -5, C3 -8 -> +8, O1 -12 -> +12: same pattern, colours swapped",
  "Correlation: +1 = same pattern, -1 = same pattern with signs flipped, 0 = unrelated",
  "So we use the **absolute value**: |+0.95| = |-0.95| = 0.95, both count as the same match",
  "In simple words: we care about the **shape** of the map, not which colour is where",
], path.join(F, "polarity.png"), { textW: 6.6 });

table("The four classes and the brain networks linked to them", [
  ["Class", "Shape on the scalp", "Network linked to it (literature)", "Our state"],
  ["A", "diagonal: left-back <-> right-front", "auditory / temporal network (verbal, phonological processing)", "state0"],
  ["B", "other diagonal: left-front <-> right-back", "visual network (occipital, visual imagery)", "state2"],
  ["C", "front <-> back along the midline", "salience network (anterior cingulate, insula); some papers: default-mode network", "state3"],
  ["D", "centre-front maximum (fronto-central)", "dorsal attention / fronto-parietal network (attention, executive control)", "state1"],
], { colW: [0.9, 3.6, 5.9, 1.7], rowH: 0.6, fs: 14,
  note: "Where the links come from: studies recording EEG and fMRI at the same time (Britz, Van De Ville & Michel 2010) correlated the moments of each microstate with fMRI networks. This is correlation evidence and still debated: say **'associated with'**, never 'C is the salience network'. Memory aid: salience = 'what is important around me?'; default mode = 'what is happening in my thoughts?' (daydreaming, remembering, self-reflection).", noteY: 4.7 });

tf("Microstates in schizophrenia: what is known", [
  "**Rieger et al. 2016** (meta-analysis): more C and less D in patients",
  "**Kim et al. 2021**, on our dataset: the same pattern (C up, D down)",
  "**Keihani et al. 2022**, on the same dataset: the **opposite** (C less, D more), as they note themselves",
  "**Ours**: C duration higher in patients (median 64.4 -> 69.2 ms, rank-sum p = 0.027 uncorrected; Holm 0.41); D lower but not significant",
  "So our trend matches Kim and the meta-analysis, but it is **tentative**:",
  "- not significant after correction for 16 tests",
  "- C and D are our least certain names (Module 6)",
], null, { qa: ["Do your results confirm the literature?", "They point the same way as Kim et al. and the meta-analysis, but they are not significant after correction, so we call them a trend."] });



/* =====================================================================================
   MODULE 1 - DATASET
   ===================================================================================== */
section(1, "The dataset", ["Nothing: this is the start"], ["28 EDF files in MLproject/dataset/", "a label per person from the file name"],
  ["dataset/*.edf (not in git)", "eeg_preprocessing.recording_files()", "eeg_preprocessing.subject_and_label()"]);

tf("RepOD 'EEG in schizophrenia'", [
  "Olejarczyk & Jernajczyk (2017), Institute of Psychiatry and Neurology, Warsaw. DOI 10.18150/repod.0107441",
  "**14 patients** with paranoid schizophrenia (s01-s14, label 1) and **14 healthy controls** (h01-h14, label 0)",
  "19 electrodes, 10-20 system, **250 Hz**, eyes closed, resting, about 15 minutes each",
  "The label comes only from the **file name**: subject_and_label('s07.edf') -> ('s07', 1)",
  "Files sorted h01 ... h14, s01 ... s14; this order matters for the random generator (seed 42) later",
  "The same dataset is used by Kim et al. 2021 and Keihani et al. 2022 (Module 13)",
], null, { qa: ["Why is 28 people a problem?", "One person = 3.6 percentage points of accuracy, so every result has a wide uncertainty. Everything is exploratory."] });

tf("Electrode names: how to read them", [
  "Letter = region: **Fp** front pole, **F** frontal, **C** central, **T** temporal, **P** parietal, **O** occipital",
  "Number = side: **odd = left** (Fp1, F3, C3 ...), **even = right** (Fp2, F4, C4 ...), **z = midline** (Fz, Cz, Pz)",
  "Old names in our files: T3, T4, T5, T6. Modern names: **T7, T8, P7, P8**. We rename them so MNE can place them",
  "19 electrodes is few: a scalp map between them is **interpolated** (filled in smoothly) when drawn",
], path.join(F, "maps_with_names.png"), { textW: 5.6 });

tf("Subject s14 is unusual: our rule for it", [
  "s14's recording is **36 minutes** long (others about 15)",
  "At about **18 minutes** its signal changes abruptly: the share of time in state C jumps by **0.415** (next largest jump of any subject: 0.191, s11)",
  "Rule, fixed **without looking at the labels**:",
  "- use the **first 15 minutes** (180 epochs) for everybody",
  "- report every result **with s14** (28 people, main) and **without s14** (27 people)",
  "Results that hold only with s14 are weaker evidence",
  "Note: the GFP peaks that **fit the maps** still come from each **whole** recording (for s14 all 36 min); only the features use the first 15 min (Module 3)",
], path.join(P, "inspect_s14.png"), { textW: 5.4,
  qa: ["Why not just delete s14?", "Deleting a patient because of their data could look like cherry-picking. Showing both is honest."] });

table("How many epochs each person keeps", [
  ["", "Epochs available (max 180)", "Kept after rejection", "Notes"],
  ["Controls (14)", "2,513", "2,378", "h14 has only 173 epochs; h04 keeps the fewest (159, 88%)"],
  ["Patients (14)", "2,465", "2,408", "s06 is short (148 epochs); s01 169, s10 170, s05 178"],
  ["Total", "4,978", "4,786", "192 rejected (3.9%)"],
], { colW: [2.2, 3.0, 2.6, 4.3], rowH: 0.5, fs: 14,
  intro: "From outputs/epoch_rejection_report_15min.csv (made by extract_epoch_features.py):",
  note: "Every person still keeps at least 141 epochs (about 12 minutes), so the per-person features are averages over many epochs. Some recordings are shorter than 15 minutes, which is why some have fewer than 180.", noteY: 4.5 });


/* =====================================================================================
   THE DATA MATRIX
   ===================================================================================== */
section(0, "The data matrix: from 19 x 225,000 to 16 numbers", ["one recording: 19 electrodes x 225,000 samples"], ["16 numbers per person", "(28 people -> a 28 x 16 table)"],
  ["all pipeline modules", "numbers: project_walkthrough/matrix_numbers_h01.csv"],
  "The faculty asked to explain the big matrix and how it gets smaller. Every number on these slides is a real count from subject h01 or from the whole dataset.");

tf("What '19 x 225,000' means", [
  "250 Hz = 250 samples per second -> one sample every 1/250 s = **4 ms**",
  "15 minutes = 15 x 60 = **900 s** -> 900 x 250 = **225,000 samples** per electrode",
  "19 electrodes -> a matrix of **19 rows (electrodes) x 225,000 columns (samples)** = **4,275,000 numbers** per person",
  "**One column = 19 numbers = one scalp map.** So one person has 225,000 scalp maps",
  "h01's file is actually 925 s = 231,250 samples; we keep the **first 15 min** (180 epochs) for everybody",
], path.join(F, "data_matrix.png"), { textW: 4.6,
  qa: ["Is it 19 x 225,000 or 225,000 x 19?", "The same data. MNE stores channels x samples (19 rows); clustering treats each map as a row (225,000 x 19)."] });

tf("One column = 19 numbers = one map", [
  "Left: the 19 voltages at one GFP peak of h01, one bar per electrode (red = positive, blue = negative)",
  "Right: the **same 19 numbers** drawn on the head: that is a scalp map",
  "Here O1 and O2 (back) are strongly negative, Fp1/Fp2/F7/F3 (front) positive: a front-back pattern (like map C)",
  "Clustering compares these 19-number patterns; the head picture is only a way to look at them",
], path.join(F, "pipeline_fig3_one_map.png"), { stack: true, textH: 1.5 });

peakTables();

tf("Why we do not cluster all 225,000 maps", [
  "Most samples have a **weak** map (low GFP): the pattern is passing through zero or changing, so it is mostly noise",
  "At **GFP peaks** the map is strongest and clearest (Module 5): about 21 per second in our data",
  "So from each person only the peak maps go into k-means, and only 1,500 of them, so everybody counts equally",
  "All 28 people x 225,000 = 6.3 million maps would also be slow, and dominated by noisy maps",
  "After the 4 maps are found, **every** sample is labelled again (Module 7): the full time course is used there",
  "Detail: the peaks are taken from the **whole cleaned recording** (h01: 925 s); the features use the first 15 minutes",
], null);

table("One person (h01): how the matrix shrinks, with real counts", [
  ["Step", "Shape", "How many numbers", "What happened"],
  ["1. raw recording", "19 x 231,250", "4,393,750", "925 s at 250 Hz"],
  ["2. cleaned", "19 x 231,250", "4,393,750", "same shape; values filtered and re-referenced"],
  ["3. first 15 min, cut into epochs", "180 x 19 x 1,250", "4,275,000", "= 19 x 225,000"],
  ["4. noisy epochs removed", "180 x 19 x 1,250", "4,275,000", "h01 loses none (h04 loses 21)"],
  ["5. GFP curve", "1 x 231,250", "231,250", "19 numbers -> 1 strength value per sample"],
  ["6. GFP peaks", "20,481 x 19", "389,139", "maps at the peaks only"],
  ["7. drop loudest 5%", "19,457 x 19", "369,683", "likely artefacts out"],
  ["8. random 1,500", "1,500 x 19", "28,500", "h01's share of the 42,000"],
  ["9. labels (after the 4 maps exist)", "180 x 1,250", "225,000", "216,525 labelled; edges = -1"],
  ["10. segments", "11,587", "", "64.4 per 5-s epoch; 11,407 transitions"],
  ["11. features per epoch", "180 x 24", "4,320", ""],
  ["12. features per person", "1 x 24 -> 1 x 16", "16", "mean over epochs; 8 redundant columns dropped"],
], { colW: [3.6, 2.5, 2.2, 3.8], rowH: 0.36, fs: 13 });

table("All 28 people together", [
  ["Step", "Shape", "How many numbers"],
  ["kept epochs of everybody (the EEG we analyse)", "4,786 x 19 x 1,250", "113,667,500"],
  ["pooled peak maps for clustering", "42,000 x 19", "798,000"],
  ["the model: 4 maps", "4 x 19", "76"],
  ["labels of every sample", "4,786 x 1,250", "5,982,500"],
  ["features per epoch", "4,786 x 24", "114,864"],
  ["features per person (24)", "28 x 24", "672"],
  ["final table", "28 x 16", "448"],
  ["classifier output", "28 predictions", "28"],
], { colW: [6.0, 3.3, 2.8], rowH: 0.45, fs: 15,
  note: "About 114 million voltage values end up as 448 numbers (16 per person). The 76 numbers of the 4 maps are the 'dictionary' that translates the EEG into a sequence of letters.", noteY: 5.6 });

flow("The shrinking matrix in one picture", [
  ["19 x 225,000", "one person, 15 min"],
  ["180 x 19 x 1,250", "cut into 5-s epochs; noisy ones out"],
  ["about 20,000 x 19", "GFP peak maps (whole recording)"],
  ["1,500 x 19", "5% cut + random choice"],
  ["42,000 x 19", "28 people pooled"],
  ["4 x 19", "modified k-means: the 4 maps"],
  ["180 x 1,250 letters", "every sample labelled A/B/C/D"],
  ["1 x 16", "per person; 28 x 16 -> classifier", true],
], { perRow: 4, bh: 1.4, rgap: 0.65 });

/* =====================================================================================
   MODULE 2 - CLEANING
   ===================================================================================== */
section(2, "Cleaning the EEG", ["one EDF file"], ["a clean recording: 19 channels, 2-20 Hz, average reference", "5-s epochs", "the list of kept epochs"],
  ["eeg_preprocessing.py: load_raw, filter_and_reference, load_clean, cut_epochs, kept_epochs", "check_bad_channels.py"],
  "Cleaning is done the same way for every person, and never uses the diagnosis.");

flow("Seven cleaning steps (eeg_preprocessing.py)", [
  ["1. Read", "mne.io.read_raw_edf(preload=True)"],
  ["2. Rename", "T3->T7, T4->T8, T5->P7, T6->P8; set_montage('standard_1020')"],
  ["3. Repair", "interpolate_bads: s07 C4, h14 F7, s11 F7, s12 P7"],
  ["4. Filter", "band-pass 2-20 Hz (raw.filter(2, 20))"],
  ["5. Re-reference", "set_eeg_reference('average')"],
  ["6. Epochs", "make_fixed_length_epochs(5 s), first 180"],
  ["7. Reject", "robust z of log-variance, |z| > 3.5 removed"],
], { perRow: 4, bh: 1.5, rgap: 0.65,
  note: "Steps 1-3 = load_raw(); 4-5 = filter_and_reference(); 1-5 together = load_clean(); 6 = cut_epochs(); 7 = kept_epochs().",
  qa: ["Why the order filter -> reference?", "Both are linear, so the order does not change the result; the code filters first, then re-references."] });

tf("Steps 1-2: read the file, fix channel names, place electrodes", [
  "**read_raw_edf** loads all 19 channels into memory (preload=True), in volts",
  "Our files use old names (T3, T4, T5, T6). MNE's standard head uses new ones, so we **rename** them",
  "**set_montage('standard_1020')** gives every electrode an x, y, z position on a standard head",
  "Positions are needed for two things:",
  "- drawing scalp maps (where to put each value)",
  "- rebuilding a broken electrode from its neighbours (step 3)",
], path.join(F, "maps_with_names.png"), { textW: 5.6,
  qa: ["What is a montage?", "The list of electrode positions on the head. 'standard_1020' is the textbook 10-20 layout."] });

tf("Step 3: find and repair broken electrodes", [
  "**check_bad_channels.py** (prints only), on a 1-40 Hz copy, three label-free checks per channel:",
  "- **flat**: variance below 5% of the median channel variance (dead electrode)",
  "- **noisy**: robust z of the variance above 5 (far louder than the rest)",
  "- **low correlation**: best |r| with any other channel below 0.3 (bad contact; neighbours always share signal)",
  "Found: **s07 C4, h14 F7, s11 F7, s12 P7** -> the BADS list in eeg_preprocessing.py",
  "**interpolate_bads** (spherical spline) rebuilds each one from the others using the head geometry",
], null, { qa: ["Why not just delete the bad channel?", "Every person must have the same 19 channels, because all are described with the same 4 maps of 19 values."],
  notes: "A bad electrode is like a broken microphone in a choir recording. We can estimate what it should have recorded from the microphones around it. That is interpolation." });

tf("How a broken electrode is rebuilt: spherical-spline interpolation", [
  "MNE: raw.interpolate_bads(), method of Perrin et al. (1989)",
  "1. The electrodes are placed on a sphere fitted to the head (their 10-20 positions)",
  "2. The bad electrode's own signal is thrown away",
  "3. At every sample, a **smooth surface** is fitted through the voltages of the 18 good electrodes; the 'spline' is the smoothest surface that passes through them",
  "4. The bad electrode gets the value of that surface **at its own position**",
  "In practice: a weighted sum of the good electrodes; weights depend on the angle between electrodes on the sphere (a Legendre-polynomial series), so near neighbours count most",
  "Only one channel per affected person (s07 C4, h14 F7, s11 F7, s12 P7), so 18 real electrodes always remain",
], null, { qa: ["Is the rebuilt channel real data?", "No: it is estimated from its neighbours. That is acceptable for one channel out of 19, because neighbouring electrodes record very similar signals."] });

tf("Step 4: band-pass filter 2-20 Hz", [
  "Keeps only rhythms between **2 and 20 Hz** (cycles per second)",
  "Removes slow drift below 2 Hz (sweat, movement) and fast noise above 20 Hz (muscle, 50 Hz mains)",
  "Alpha (8-12 Hz), the main eyes-closed rhythm, is inside the band",
  "**Why 2-20?** It is the band used for microstates in Kim et al. [9] and Keihani et al. [10] on this same dataset",
  "Code: raw.filter(2, 20): MNE's default zero-phase FIR filter (no time shift)",
], path.join(F, "psd_before_after.png"), { textW: 5.3,
  caption: "x = frequency (Hz), y = power (log scale). Outside the shaded band the power drops.",
  qa: ["What is on the axes of this plot?", "x = frequency in Hz; y = power, how much of the signal is at that frequency (log scale)."] });

tf("The filter on one electrode (Cz, h01, 2 seconds)", [
  "Grey = before the filter, black = after 2-20 Hz: the fast jagged bits (above 20 Hz) are gone, the 10 Hz rhythm stays",
  "500 samples in 2 s: one every 4 ms. Same time, same electrode; only the frequencies outside 2-20 Hz were removed",
], path.join(F, "pipeline_fig1_filter.png"), { stack: true, textH: 1.0 });

tf("What cleaning does to the signal (same 3 s of h01)", [
  "Left: as recorded. Right: after 2-20 Hz and average reference",
  "Slow wandering and sharp noise are gone; the rhythm stays",
], path.join(F, "raw_vs_clean.png"), { stack: true, textH: 0.85 });

tf("Step 5: average reference", [
  "EEG measures voltage **relative to a reference electrode**. Change the reference and every value changes",
  "Average reference: at every sample, **subtract the mean of the 19 channels** from each channel",
  "Worked example with 3 electrodes: values 10, 4, 1 -> mean 5 -> new values **5, -1, -4** (they now sum to 0)",
  "Why: a microstate map is a **pattern across electrodes**. It must not depend on which electrode was the reference",
  "Side effect we use later: the mean across channels is 0, so **GFP = standard deviation across the 19 channels**",
], null, { qa: ["Does average reference change the shape of the map?", "No: it shifts all 19 values by the same amount at each instant. The pattern (differences between electrodes) stays."] });

tf("Steps 6-7: cut into epochs and reject noisy ones", [
  "**cut_epochs**: 5-s non-overlapping pieces, first 180 kept (15 min)",
  "**kept_epochs**: for each epoch, log of its variance (how loud it is)",
  "- robust z = 0.6745 x (value - median) / MAD",
  "- MAD = median absolute deviation; 0.6745 makes it comparable with an SD",
  "- **|z| > 3.5 -> rejected** (blinks, movement, flat stretches)",
  "Robust (median/MAD), because a mean and SD would be pulled by the bad epochs themselves",
  "Done per person, no labels. Result: **4,786 of 4,978** epochs kept (88-100% per person); Keihani et al. used a similar rule (variance > 3 SD)",
], path.join(F, "epoch_rejection.png"), { textW: 5.0,
  qa: ["Why 5-second epochs?", "Same as [9] and [10]. Long enough for about 60-80 microstate segments, short enough to drop a bad stretch without losing much."] });


/* =====================================================================================
   MODULE 3 - GFP
   ===================================================================================== */
section(3, "GFP and GFP peaks", ["a clean recording (19 x samples)"], ["GFP curve (1 number per sample)", "the maps at GFP peaks", "1,500 peak maps per person"],
  ["microstate_model.py: global_field_power, sample_gfp_peaks", "pycrostates: extract_gfp_peaks", "notebooks/toy_gfp_demo.ipynb"],
  "This is the module the faculty asked about most: what GFP is, why peaks, how they are computed, which library, and what is on the axes.");

tf("What is GFP (global field power)?", [
  "At each sample, take the 19 channel values. GFP = **how spread out they are**:",
  "- GFP(t) = sqrt( (1/19) x sum over channels of (V_i(t) - mean V(t))^2 )",
  "- after average reference the mean is 0, so **GFP = standard deviation across channels**",
  "Code: global_field_power(data) = data.std(axis=0)  (data = channels x samples)",
  "High GFP = strong, clear scalp pattern. Low GFP = flat scalp, pattern weak or changing",
  "**Axes of a GFP plot: x = time, y = GFP in µV.** It is not a frequency plot",
  "One channel alone has GFP = 0 (spread of one number is 0): GFP needs a pattern across electrodes",
], path.join(F, "toy_1.png"), { textW: 6.4, caption: "Toy Part 1: one channel, one 10 Hz sine -> GFP is 0 everywhere.",
  notes: "Imagine 19 people standing at different heights. GFP is how different their heights are at one moment. If everybody is at the same height, GFP is zero, even if they all jump together." });

tf("Toy Part 2: one 10 Hz wave, 19 channels, a fixed front-back pattern", [
  "Each channel = 10 Hz sine x a fixed weight (front positive, back negative)",
  "GFP = |sine| x spread of the weights -> **2 peaks per cycle** (top and bottom) -> 10 Hz gives **20 peaks per second**",
  "find_peaks and pycrostates both find 20",
], path.join(F, "toy_2.png"), { textW: 4.6,
  qa: ["Why 20 peaks for a 10 Hz wave?", "GFP ignores the sign: the map is strongest at the top AND at the bottom of each cycle, so 2 peaks per cycle."] });

tf("Toy Part 2: maps at peaks, and why we only use peaks", [
  "Consecutive peaks show the **same map with colours flipped** (top vs bottom of the wave): this is why polarity is ignored",
  "At a GFP minimum the map is mostly noise",
  "Right: |correlation with the true map| against GFP. **High GFP = clean map** (|r| near 1)",
  "So we cluster only the maps at GFP peaks: highest signal-to-noise",
], path.join(F, "toy_3.png"), { stack: true, textH: 1.7 });

tf("Toy Part 3: 50 + 100 + 200 + 300 Hz mixed", [
  "Sampled at 1,000 Hz (must be above 2 x 300 = 600 Hz, Nyquist)",
  "**FFT** (frequency axis): 4 spikes at 50, 100, 200, 300 Hz: it tells WHICH frequencies are present",
  "**GFP** (time axis): one curve over time: it tells WHEN the map is strong. 39 peaks in 100 ms",
  "Lesson: GFP is a time measure, not a frequency analysis. Our real EEG is 2-20 Hz, so peaks are much rarer",
], path.join(F, "toy_4.png"), { textW: 5.2,
  qa: ["Where does the sine data come from?", "We generate it in the notebook with numpy (np.sin) and the real electrode positions. It is made up, to understand the method."] });

tf("How the peaks are found: pycrostates extract_gfp_peaks", [
  "Call: extract_gfp_peaks(raw, min_peak_distance=2)",
  "Inside (pycrostates 0.6.1 source):",
  "- GFP = np.std over channels at every sample",
  "- scipy.signal.find_peaks(gfp, distance=2): a peak = a sample larger than its neighbours, at least 2 samples (8 ms) apart",
  "- returns a ChData object: the **19-channel map at every peak**",
  "Docs: pycrostates.readthedocs.io -> API -> preprocessing.extract_gfp_peaks",
  "Toy check: on the 10 Hz wave, plain scipy find_peaks and pycrostates give the same 20 peaks",
], null, { qa: ["Which library and why?", "pycrostates (built on MNE) for microstates: GFP peaks, modified k-means, backfitting, in one tested package."] });

tf("Toy Part 5: the same on 2 s of real EEG (h01)", [
  "Real cleaned EEG: 42 peaks in 2 s (about 21 per second, close to the 10 Hz alpha rhythm x 2)",
  "Maps at peaks look like the toy maps: smooth patterns that hold for a while, then change",
], path.join(F, "toy_7.png"), { stack: true, textH: 1.0 });

tf("Topomaps at the real peaks", [
  "Several peaks in a row share one pattern (one microstate), sometimes with flipped colours; then the pattern changes",
], path.join(F, "toy_8.png"), { stack: true, textH: 0.7 });

tf("GFP peaks on real EEG, before and after the filter", [
  "Every grey dot is one sample (4 ms apart); red dots are the GFP peaks (find_peaks, at least 2 samples apart)",
  "**Before filtering** the GFP is jagged: small bumps count as peaks. All subjects: about **45 peaks per second**, about 21 ms apart",
  "**After 2-20 Hz** the GFP is a row of smooth hills: about **21 peaks per second**, about 44 ms apart (two per alpha cycle)",
  "So filtering first is what makes the peaks meaningful: each peak is the top of a real hill, not noise",
], path.join(F, "pipeline_fig2_gfp.png"), { textW: 4.6 });

tf("From all peaks to 1,500 per person (sample_gfp_peaks)", [
  "h01 has **20,481** GFP peaks in 925 s",
  "1. Drop the **loudest 5%** (likely artefacts: blinks, jumps)",
  "2. Pick **1,500 at random**, without replacement",
  "One random generator, **seed 42**, used in file order (h01 ... s14), so the result is reproducible",
  "Same number from everybody, so **no person dominates** the maps",
  "28 x 1,500 = **42,000 peak maps** of 19 numbers go to clustering",
], path.join(F, "peak_histogram.png"), { textW: 5.4,
  qa: ["Why not use all peaks?", "Equal weight per person, and speed. 1,500 is enough: refitting with other seeds or subjects gives nearly identical maps (Module 6)."] });


/* =====================================================================================
   MODULE 4 - MAPS
   ===================================================================================== */
section(4, "Finding the 4 maps", ["42,000 peak maps (19 numbers each)", "no labels"], ["4 maps of 19 numbers (the model)", "GEV = 0.674", "names A, B, C, D"],
  ["microstate_model.py: fit_shared_model, suggest_state_names", "tune_microstates_v3.py", "gev_by_k.py, check_map_stability.py", "name_states.py, plot_labelled_maps.py"]);

tf("Ordinary k-means first (you know it from Lab 3)", [
  "Choose k centres at random",
  "1. Assign each point to the **nearest** centre (Euclidean distance)",
  "2. Move each centre to the **mean** of its points",
  "Repeat 1-2 until nothing changes",
  "Problem for EEG maps:",
  "- a map and its **sign-flipped copy** are the same brain state, but far apart in distance",
  "- the mean of a map and its flipped copy is **zero**",
  "- a strong and a weak version of the same map would be split by distance",
], null, { notes: "Ordinary k-means groups points by distance. For scalp maps, distance is the wrong measure, because the same pattern appears with both signs and with different strength." });

tf("Modified k-means (pycrostates ModKMeans)", [
  "Same loop, two changes:",
  "- **assign** by the largest **|spatial correlation|** (absolute value: polarity ignored; correlation: strength ignored)",
  "- **update** each map as the **first eigenvector** of its maps (the main direction), not the mean, so A and -A do not cancel",
  "Repeated from **50 random starts** (n_init=50); the run with the **highest GEV** is kept",
  "Code: ModKMeans(n_clusters=4, n_init=50, random_state=42).fit(pooled)",
  "Input: the 42,000 peak maps; output: model.cluster_centers_ = 4 x 19",
  "No diagnosis label is used",
], null, { qa: ["Why the absolute value of correlation?", "EEG oscillates: the same brain source shows up as a map and its flipped copy within half a cycle (see Toy Part 2)."] });

tf("GEV: how good are the 4 maps?", [
  "GEV = sum over t of (GFP(t) x corr(map at t, its template))^2 / sum over t of GFP(t)^2",
  "= share of the signal's variance explained by the templates, **weighted by GFP** (strong moments count more)",
  "Ours: **GEV = 0.674** (67.4%)",
  "GEV always rises with more maps. k = 2...8: 60.7, 64.6, **67.4**, 69.1, 70.5, 71.7, 72.7%",
  "Gain per extra map: 4.0, 2.8, then 1.6, 1.5, 1.2, 1.0 points: the gain roughly halves after 4 (a mild elbow)",
  "We use k = 4 because the literature uses the 4 classic maps A-D (incl. [9], [10]); the curve does not contradict it",
], path.join(P, "gev_by_k.png"), { textW: 6.0,
  qa: ["Why exactly 4 maps?", "Standard in the literature (comparable with other studies), and GEV 67.4% is typical. After 4 the gain per extra map roughly halves (a mild elbow, not a sharp one)."] });

table("Why exactly 4 maps: five reasons", [
  ["Reason", "In short"],
  ["1. Convention and comparability", "A-D are found in many studies; Kim et al. and Keihani et al. both used 4 on this dataset, so our results can be compared"],
  ["2. GEV levels off", "67.4% at k = 4; each extra map adds only 1.0-1.6 points; going from 4 to 8 maps doubles the states for about 5 more points"],
  ["3. More maps = more features", "features = 3k + k(k-1): k = 4: 24 (16 free); k = 5: 35 (25); k = 6: 48 (36); k = 8: 80 (64). With 28 people, more features = more overfitting"],
  ["4. Less reliable features", "an epoch has about 64 segments (h01); 8 states give 56 possible transitions instead of 12, many never seen in an epoch, so their probabilities are noisy"],
  ["5. Familiar, meaningful patterns", "with 2-20 Hz and k = 4 the four textbook A-D shapes appear clearly (earlier tuning rounds tried k = 3-6 and two frequency bands)"],
], { colW: [3.3, 8.8], rowH: 0.7, fs: 13,
  note: "k is a setting we choose (a hyperparameter); the data do not decide it on their own. The goal is not the highest GEV: one map per sample would explain everything and teach nothing.", noteY: 6.15 });

tf("Can there be more than 4 microstates?", [
  "Yes: some studies find 5, 6 or 7; with more states the classic maps split into sub-patterns",
  "Our code can do it: change N_STATES in microstate_model.py (gev_by_k.py already fits k = 2-8)",
  "But a different k changes everything after it: the maps, the A-D names, all features and all results",
  "So the project keeps 4; a different k is possible future work",
], null);

tf("Does the method work? Toy Part 4 (known answer)", [
  "Made-up 30 s signal: 4 hidden maps, switching every 40-120 ms, 10 Hz strength, plus noise",
  "1,577 peaks -> modified k-means -> all 4 maps found again with **|r| = 1.00**; **94.9%** of samples labelled correctly",
  "Mean durations: true A 79.5, B 79.1, C 71.7, D 78.3 ms vs found 80.0, 79.5, 70.2, 78.8 ms",
], path.join(F, "toy_5.png"), { textW: 4.9 });

tf("Toy Part 4: the found sequence matches the true one", [
  "Top: GFP with each peak coloured by its true state. Middle: true sequence. Bottom: what the method found",
  "The white bit at the start is unlabelled on purpose (cut by the start; our epochs do the same)",
], path.join(F, "toy_6.png"), { stack: true, textH: 1.0 });

tf("Are our maps stable?", [
  "**check_map_stability.py**: refits with 4 other seeds and with 4 random subjects left out; each new map matched to the saved one (Hungarian method), |r| printed. Rule: >= 0.9 stable",
  "Stronger evidence (Module 11): the maps refitted 55 times without one subject (28 with s14 + 27 without) match the main maps with **|r| >= 0.996** every time",
  "So the maps are not an accident of one fit, and the test subject's EEG barely changes them",
], null, { qa: ["What is the Hungarian method?", "An algorithm that finds the best one-to-one pairing (here, new map to old map by |r|). scipy.optimize.linear_sum_assignment."] });

table("Map stability: the actual numbers (check_map_stability.py)", [
  ["Refit", "Left out", "GEV", "|r| A", "|r| D", "|r| B", "|r| C"],
  ["seed 1, all 28", "-", "0.674", "1.00", "1.00", "1.00", "1.00"],
  ["seed 2, all 28", "-", "0.674", "1.00", "1.00", "1.00", "1.00"],
  ["seed 3, all 28", "-", "0.674", "1.00", "1.00", "1.00", "1.00"],
  ["seed 4, all 28", "-", "0.674", "1.00", "1.00", "1.00", "1.00"],
  ["subset 1", "h10, s07, s12, s13", "0.683", "1.00", "1.00", "1.00", "1.00"],
  ["subset 2", "h09, s03, s06, s13", "0.664", "1.00", "1.00", "1.00", "1.00"],
  ["subset 3", "h04, s02, s11, s12", "0.685", "1.00", "1.00", "1.00", "1.00"],
  ["subset 4", "h03, h09, s01, s11", "0.671", "1.00", "1.00", "1.00", "1.00"],
], { colW: [2.2, 3.3, 1.2, 1.35, 1.35, 1.35, 1.35], rowH: 0.42, fs: 14,
  note: "Each refit uses 30 random starts; every new map is paired with the saved map by the Hungarian method and |r| is printed (rule: >= 0.9 stable). All 32 matches are 1.00 (to 2 decimals): a different random start or 4 fewer people gives the same 4 maps. GEV moves only with the people included (0.664-0.685).", noteY: 5.4 });

tf("One shared model, not one per person or per group", [
  "The 42,000 peak maps of all 28 people are pooled and clustered **once**",
  "**Unsupervised**: the algorithm never sees who is a patient; it only groups maps by shape",
  "So 'state C' is the **same reference map for everybody**, and durations and transitions can be compared fairly",
  "Separate maps for patients and controls would (a) use the labels and (b) risk comparing slightly different 'C's",
  "Saved as outputs/shared_microstate_model_v3.pkl (v3 = third tuning round)",
  "For the classifiers this model is refitted inside every fold without the test person (Module 11)",
], null, { qa: ["Why not fit maps per group?", "Then 'C' in patients and 'C' in controls might be different maps, and choosing the maps would use the diagnosis."] });

tf("Naming the maps A, B, C, D (name_states.py)", [
  "k-means numbers its clusters 0-3 in random order. The literature names 4 classic shapes:",
  "- **A**: left-back <-> right-front diagonal",
  "- **B**: left-front <-> right-back diagonal",
  "- **C**: front <-> back",
  "- **D**: centre-front peak vs the edges",
  "Each of our maps is correlated (|r|) with idealised versions of these shapes; best one-to-one assignment",
  "Result: **state0 = A, state2 = B, state3 = C, state1 = D**. Scores: A 0.95, B 0.93, C 0.94 (clear); state1 scores C 0.90 vs D 0.91, so C and D are the least certain names",
  "Only the names on figures change; tables keep state numbers",
], path.join(P, "maps_labelled_ABCD.png"), { textW: 5.6 });

tf("How to read a microstate map (asked by the faculty)", [
  "Colour = voltage at each electrode **relative to the average** at that instant: red above, blue below; between electrodes it is interpolated",
  "**Polarity does not matter**: red and blue swapped is the same state (right)",
  "What matters is the **orientation of the gradient**: which areas are opposite to which",
  "Say: 'map C has a front-to-back orientation'",
  "**Do NOT say** 'more frontal activity'. A map is the field on the scalp; many different brain sources can produce the same map (inverse problem). It does not show where activity is or how much",
  "Literature links (e.g. C with salience/default-mode networks) come from fMRI studies: cite them as literature, not as our finding",
], path.join(F, "polarity.png"), { textW: 7.2,
  notes: "A map is like a photo of the weather map at one instant: high pressure here, low pressure there. It tells us the shape of the field, not which factory produced the heat." });

tf("The 4 maps with electrode names", [
  "Read one: in **C** (state 3) frontal electrodes (Fp1, Fp2, F3, F4, Fz) are on one side, occipital/parietal (O1, O2, P3, P4) on the other",
], path.join(F, "maps_with_names.png"), { stack: true, textH: 0.8,
  qa: ["Is our A really the textbook A?", "Assigned by shape similarity to idealised shapes; A and B are clear, C and D less certain. We state this."] });


/* =====================================================================================
   MODULE 5 - LABELLING AND FEATURES
   ===================================================================================== */
section(5, "Labelling and features", ["the 4 maps", "every kept epoch (19 x 1,250)"], ["a state for every sample", "24 features per epoch", "28 x 24 -> 28 x 16 table"],
  ["microstate_model.label_epoch (model.predict)", "microstate_features.py (formulas)", "extract_epoch_features.py", "make_final_features.py", "plot_transition_matrix.py, make_epoch_array.py"]);

tf("Labelling one sample: which map is closest?", [
  "Take the 19 values at one sample; correlate with each of the 4 maps; **largest |r| wins** (sign ignored)",
  "Example (h01, epoch 12, t = 1,276 ms): C +0.89 (closest), A +0.80, D -0.78, B -0.35",
], path.join(F, "assign_one_sample.png"), { stack: true, textH: 1.0,
  qa: ["Why every sample, not only peaks?", "Peaks were used to FIND the maps. To describe how long each state lasts we need the state at every moment."] });

tf("Labelling a whole epoch: backfitting + smoothing", [
  "Without smoothing the label flickers: **120 segments in 2 s**. Smoothed (factor 5, half window 5 = 11 samples = 44 ms, Pascual-Marqui 1995): **32 segments**",
  "Smoothing changes 36% of the samples' labels in this epoch: e.g. the sample above (C, but A was close at 0.80) becomes A, like its neighbours",
  "reject_edges=True: the first and last segment become -1 (grey), as their true length is unknown",
], path.join(F, "backfitting_epoch.png"), { stack: true, textH: 1.45 });

tf("How the smoothing works (Pascual-Marqui et al. 1995, inside pycrostates)", [
  "For every sample t and every map k, a **cost**:",
  "- cost(k, t) = misfit(k, t) - b x N_k(t)",
  "- misfit = how badly map k explains the 19 values at t: (|V_t|^2 - (map_k . V_t)^2) / (2 e (19 - 1)), with e = the average leftover noise",
  "- N_k(t) = how many samples in the window t-5 ... t+5 (11 samples = 44 ms) currently have label k",
  "- b = smoothing factor = **5**: each neighbour with the same label is a discount",
  "Each sample takes the map with the **lowest cost**; the window counts change, so this is **repeated** until the total misfit stops changing (tolerance 1e-5)",
  "Effect: a sample keeps a lone label only if that map fits clearly better; if two maps fit almost equally, the neighbours decide",
  "Code: model.predict(epoch, factor=5, half_window_size=5, reject_edges=True) in microstate_model.label_epoch",
], null, { qa: ["Doesn't smoothing invent data?", "No: it only chooses between maps that fit almost equally well. It removes one-sample flickers that are noise, not changes of state."] });

tf("1 second of labels becomes segments", [
  "The black line is GFP; the background colour is the map each sample was given (after smoothing)",
  "In this 1 second there are **21 segments**: D B D A D C A D C B D C D C D A D C A D A",
  "Each segment = one visit to a state. Its length gives the **duration**, the number of visits the **occurrence**, the time share the **coverage**, and the order (D -> B -> D -> A ...) the **transitions**",
], path.join(F, "pipeline_fig4_labels.png"), { stack: true, textH: 1.4 });

tf("Segments and the 3 state features (microstate_features.py)", [
  "**Segment** = run of the same state: A A A B B C C C C -> (A, 3 samples), (B, 2), (C, 4)",
  "fs = 250 Hz; L = labelled samples in the epoch (edges excluded); n_k = number of segments of state k; l_kj = length of its j-th segment",
  "- **Duration_k** = (1/n_k) x sum_j l_kj / fs: mean length of one visit (seconds)",
  "- **Occurrence_k** = n_k / (L / fs): visits per second",
  "- **Coverage_k** = sum_j l_kj / L: share of the time (0 to 1)",
  "So **Coverage = Occurrence x Duration**, exactly, within an epoch",
  "Same definitions as pycrostates 0.6.1 (checked: scripts/check_features_match.py, 345 epochs, identical) and as Keihani et al. (Table 1)",
  "Per person: mean over the kept epochs; an epoch where a state never appears is skipped for its duration (its occurrence and coverage count as 0)",
], null, { qa: ["Typical values in our data?", "Means over the 28 people: duration 63-69 ms, occurrence 3.6-4.2 per s, coverage 22-28% per state. Keihani et al. report 56-80 ms on the same data."] });

table("Worked example (computed by our code)", [
  ["State", "Segments (samples)", "Duration", "Occurrence", "Coverage", "Check: occ x dur"],
  ["A", "15, 12 -> n = 2", "(15+12)/2/250 = 0.054 s", "2 / 0.4 s = 5.0 /s", "27/100 = 0.27", "5.0 x 0.054 = 0.27"],
  ["B", "20, 10 -> n = 2", "15/250 = 0.060 s", "5.0 /s", "0.30", "0.30"],
  ["C", "18 -> n = 1", "0.072 s", "2.5 /s", "0.18", "0.18"],
  ["D", "25 -> n = 1", "0.100 s", "2.5 /s", "0.25", "0.25"],
], { colW: [0.9, 2.3, 2.8, 2.2, 1.8, 2.1], rowH: 0.45, fs: 14,
  intro: "Made-up epoch: 3 unlabelled, A x15, B x20, C x18, A x12, D x25, B x10, 4 unlabelled -> L = 100 labelled samples = 0.4 s",
  note: "Transitions of the same epoch: without repeats **A B C A D B** -> pairs A->B, B->C, C->A, A->D, D->B. Out of A there are 2 (to B and to D): P(A->B) = 0.5, P(A->D) = 0.5. Out of B there is 1 (to C): P(B->C) = 1.", noteY: 4.5 });

tf("Transition probabilities", [
  "Remove repetitions: A A A D D A C -> **A D A C** (staying in a state is not a transition)",
  "c_ij = number of times i is directly followed by j; **P(i -> j) = c_ij / all transitions out of i**",
  "4 states x 3 targets = **12 transitions**; each row sums to 1",
  "Per person: counts of all kept epochs **added first, then divided** (pooled): one epoch has only about 63 transitions (h01: 11,407 in 180 epochs), so per-epoch probabilities are noisy",
  "It is a first-order ('jump') transition matrix, used as a **description**; we do not assume the sequence is truly Markov (von Wegner et al. 2017: resting sequences have a longer memory)",
  "Edge segments (-1) are skipped",
], path.join(P, "transition_matrix_heatmap.png"), { stack: true, textH: 3.0,
  caption: "Row = from, column = to. Controls, patients, and patients minus controls (descriptive, no test)." });

tf("What our transition matrix shows (descriptive only)", [
  "From A, B and D the most likely next state is **C** (0.36-0.39, against 1/3 if equal); from C the most likely next is **D** (0.38 in controls). C also has the largest coverage (27.7%)",
  "Patients switch **A -> B** more often: 0.307 -> 0.346 (+0.04). This is trans_0to2: the strongest rank-sum trend, and kept by L1 in **every** fold",
  "Patients switch **into D** less from A, B and C (about -0.03 each), in line with D being lower in patients",
  "Duration/occurrence/coverage say **how much** each state is present; transitions say the **order** of switching (the 'syntax'). Altered syntax in schizophrenia: Lehmann et al. 2005",
  "All small and uncorrected: describe, do not claim",
], path.join(P, "transition_matrix_heatmap.png"), { stack: true, textH: 2.5 });

flow("From epochs to one row per person (extract_epoch_features.py)", [
  ["For each person", "load_clean -> cut_epochs -> kept_epochs"],
  ["For each kept epoch", "label_epoch -> epoch_features: 24 numbers + counts"],
  ["Per person", "mean of the epochs (states); pooled counts (transitions)"],
  ["Save 4 tables", "epoch table (4,786 rows), mean (28 x 24), SD, rejection report", true],
], { perRow: 4, bh: 1.8, top: 1.7,
  note: "A state that never appears in an epoch gets duration = NaN there, and nanmean ignores it. Epoch rows of one person are NOT independent: always split by subject, never by row.",
  qa: ["Mean or median across epochs?", "Mean (nanmean) for the 12 state features; pooled counts for the 12 transitions."] });

tf("24 -> 16 features (make_final_features.py): drop the redundant ones", [
  "**Drop the 4 coverages**: coverage = duration x occurrence, and the 4 coverages sum to 1",
  "**Drop 1 transition per source state**: the 3 out of a state sum to 1, so the third is determined by the other two",
  "- fixed rule: drop the LAST column of each group: 0->3, 1->3, 2->3, 3->2",
  "**Keep 16**: 4 durations + 4 occurrences + 8 transitions",
  "**No labels used** -> cannot inflate accuracy",
  "Writes features_final_16.csv (28), features_final_16_no_s14.csv (27), features_final_16_list.txt",
], null, { qa: ["Is dropping features before cross-validation leakage?", "No: the rule uses only the structure of the features (sums to 1), never the labels or any statistic of the data."] });

table("The 16 features (and their names)", [
  ["Column", "Meaning", "Column", "Meaning"],
  ["state0_duration", "A duration (s)", "trans_0to1", "A -> D"],
  ["state0_occurrence", "A occurrence (per s)", "trans_0to2", "A -> B"],
  ["state1_duration", "D duration", "trans_1to0", "D -> A"],
  ["state1_occurrence", "D occurrence", "trans_1to2", "D -> B"],
  ["state2_duration", "B duration", "trans_2to0", "B -> A"],
  ["state2_occurrence", "B occurrence", "trans_2to1", "B -> D"],
  ["state3_duration", "C duration", "trans_3to0", "C -> A"],
  ["state3_occurrence", "C occurrence", "trans_3to1", "C -> D"],
], { colW: [2.8, 3.25, 2.8, 3.25], rowH: 0.42, fs: 14,
  note: "Remember: state0 = A, state1 = D, state2 = B, state3 = C. Durations are in seconds in the CSV (0.060 = 60 ms).", noteY: 5.4 });

tf("make_epoch_array.py: the raw epochs, saved for later", [
  "The feature tables keep only summaries. Some later steps need the **EEG itself**:",
  "- deep learning (LaBraM, Module 12)",
  "- refitting the maps inside each fold (Module 11): every person must be labelled again with each fold's maps",
  "Same cleaning and the **same 4,786 kept epochs** as the features -> outputs/epochs_15min.npz (420 MB, not in git)",
  "--preset labram: the same epochs, 0.1-75 Hz, 50 Hz notch, resampled to 200 Hz (LaBraM's training setting)",
], null);


/* =====================================================================================
   MODULE 6 - GROUP STATISTICS
   ===================================================================================== */
section(6, "Group statistics (HC vs SC)", ["the 28 x 16 table", "the labels"], ["a p-value per feature (with and without correction)", "figures and a Kim-style table"],
  ["ranksum_group_test.py", "group_specific_analysis.py", "make_kim_style_tables.py", "plot_group_comparison.py, plot_group_maps_labelled.py"]);

tf("Wilcoxon rank-sum test (= Mann-Whitney U)", [
  "Question per feature: are patients' values generally higher or lower than controls'?",
  "1. Put all 28 values in order and give ranks 1 ... 28",
  "2. Add up the patients' ranks",
  "3. Compare with what random group labels would give -> z and p",
  "No bell-curve assumption: fits 14 vs 14 with outliers",
  "Effect size r = z / sqrt(28): 0.1 small, 0.3 medium, 0.5 large",
  "Code: scipy.stats.ranksums",
], null, { qa: ["Why not a t-test?", "With 14 per group we cannot trust normality; the rank-sum test only uses the order of the values. The faculty said rank-sum is enough."] });

tf("Many tests at once: Holm correction", [
  "16 tests at p < 0.05: even if nothing is real, about **16 x 0.05 = 0.8 false positives** are expected",
  "Holm: sort the 16 p-values; multiply the smallest by 16, the next by 15, ...; keep the result non-decreasing",
  "We report both: **p_individual** (each feature on its own) and **p_holm** (corrected)",
  "A feature 'differs' only if p_holm < 0.05",
], null);

tf("Result: trends only, nothing survives correction", [
  "Shared maps (the main maps):",
  "- **C duration**: HC 64.4 vs SC 69.2 ms, p = 0.027, r = 0.49 -> Holm p = 0.41",
  "- **A -> B**: HC 0.302 vs SC 0.346, p = 0.019, r = 0.52 -> Holm p = 0.31",
  "- all other 14: p > 0.14",
  "**Wording**: 'C duration and A->B tend to be higher in patients (uncorrected p < 0.05), but no feature differs after correction for 16 tests'",
  "Never: 'patients have longer C'",
], path.join(P, "group_specific", "ranksum_HC_SC_shared_maps.png"), { textW: 5.2 });

table("Rank-sum test, all 16 features (shared maps)", [
  ["Feature", "HC median", "SC median", "Higher in", "p", "Holm p", "r", "Feature", "HC median", "SC median", "Higher in", "p", "Holm p", "r"],
  ["A duration", "59.5 ms", "62.1 ms", "SC", "0.27", "1.00", "0.25", "A -> D", "0.357", "0.295", "HC", "0.29", "1.00", "-0.24"],
  ["A occurrence", "3.32", "3.73", "SC", "0.82", "1.00", "0.05", "A -> B", "0.302", "0.346", "SC", "0.019", "0.31", "0.52"],
  ["D duration", "72.3 ms", "64.1 ms", "HC", "0.52", "1.00", "-0.14", "D -> A", "0.269", "0.284", "SC", "0.78", "1.00", "0.06"],
  ["D occurrence", "3.86", "3.71", "HC", "0.14", "1.00", "-0.33", "D -> B", "0.312", "0.313", "SC", "0.85", "1.00", "0.04"],
  ["B duration", "59.9 ms", "64.9 ms", "SC", "0.14", "1.00", "0.33", "B -> A", "0.295", "0.331", "SC", "0.38", "1.00", "0.19"],
  ["B occurrence", "3.85", "3.82", "HC", "0.68", "1.00", "0.09", "B -> D", "0.355", "0.294", "HC", "0.38", "1.00", "-0.19"],
  ["C duration", "64.4 ms", "69.2 ms", "SC", "0.027", "0.41", "0.49", "C -> A", "0.302", "0.339", "SC", "0.68", "1.00", "0.09"],
  ["C occurrence", "4.09", "4.03", "HC", "0.78", "1.00", "-0.06", "C -> D", "0.410", "0.339", "HC", "0.23", "1.00", "-0.27"],
], { colW: [1.25, 0.85, 0.85, 0.7, 0.65, 0.65, 0.6, 1.0, 0.85, 0.85, 0.7, 0.75, 0.65, 0.65], rowH: 0.42, fs: 11,
  note: "From outputs/group_specific/ranksum_results.csv. Occurrence in visits per second, transitions as probabilities. Only two features are below 0.05 on their own; after Holm correction for 16 tests none is (smallest Holm p = 0.31). r = effect size (+ = higher in SC).", noteY: 5.4 });

tf("The same comparison as box plots", [
  "One small chart per feature: every dot is one person (14 HC, 14 SC); the box is the middle half of the group, the line the median; the title gives the Holm-corrected p",
], path.join(P, "group_specific", "group_comparison_HC_SC.png"), { stack: true, textH: 0.7 });

tf("Group-specific maps (asked by the faculty)", [
  "Maps fitted **separately** for HC and SC (1,500 peaks each, k = 4, 50 starts); each group map gets the letter of its most similar shared map",
  "Features recomputed with each group's own maps",
  "Rank-sum: D occurrence p = 0.031, A -> D p = 0.009 (Holm 0.14): again nothing survives",
  "For group comparison only, **never for classification** (it would need the label to choose the maps)",
  "The faculty later said the shared-map approach with LOSO is the main one",
], path.join(P, "group_specific", "group_maps_HC_SC_labelled.png"), { textW: 5.0 });

tf("Kim et al. style table", [
  "Same layout as Table 2 of Kim et al. 2021 [9]: mean (SD) per group, Cohen's d, p, t for duration, occurrence, coverage of A-D",
  "Kim et al.: C up and D down in patients. Keihani et al.: the opposite. Ours: C duration up (trend), D not significant",
], path.join(P, "group_specific", "kim_style_table_shared_maps.png"), { stack: true, textH: 0.8 });


/* =====================================================================================
   MODULE 7 - EVALUATION
   ===================================================================================== */
section(7, "How we evaluate", ["a feature table (28 rows, or many rows per person)", "a model"], ["one prediction per person", "accuracy, 95% CI, other metrics, train accuracy"],
  ["evaluate.py: evaluate, wilson_ci, subject_probabilities", "labs/lab_common.py"],
  "How to explain the train/test split: we pretend one person is new. We train on everyone else, and test on that person. We do this for each of the 28 people, so every person is tested exactly once, by a model that never saw them.");

tf("Why split by person, never by epoch", [
  "Epochs of one person are very similar to each other (same head, same electrodes, same day)",
  "If epochs of the same person are in train and test, the model **recognises the person**, not the disease",
  "That is why some papers on this dataset report 90-99%: they split epochs or augment the data (Module 13)",
  "Racz & Csukly 2025 re-ran pipelines on this dataset: leaky vs leakage-free gap of **28 points**",
  "**Our rule**: all epochs of a person are always on the same side of every split",
], null, { qa: ["Then why do we have 4,786 epochs but only 28 test cases?", "Because the unit we want to predict is a person. Epochs help describe a person, but they are not independent cases."] });

flow("Leave-one-subject-out (LOSO), one fold", [
  ["1. Hold out", "1 person = test (e.g. h01)"],
  ["2. Fit on 27", "scaler + model learn from the other 27 only"],
  ["3. Predict", "P(patient) for the held-out person (average over epochs if many rows)"],
  ["4. Repeat x 28", "every person tested once -> 28 predictions -> accuracy", true],
], { perRow: 4, bh: 1.7, top: 1.6,
  note: "StandardScaler is inside the pipeline, so its mean and SD come from the 27 training people only. With the 27-person table (without s14) there are 27 folds.",
  qa: ["How do you explain the train/test split?", "Train = 27 people, test = 1 person, 28 times. Every number we report is from people the model had not seen."] });

table("What evaluate() reports", [
  ["Number", "Meaning"],
  ["accuracy", "share of the 28 people predicted correctly (P(patient) >= 0.5 -> patient)"],
  ["95% CI (Wilson)", "range of accuracies consistent with the data; e.g. 23/28 = 82.1% -> 64.4% to 92.1%"],
  ["balanced accuracy", "mean of recall for patients and for controls (fair if classes are unequal, e.g. 14 vs 13)"],
  ["precision / recall / F1", "of those called patient, how many are; of the patients, how many are found; their balance"],
  ["AUC", "how well P(patient) ranks patients above controls (0.5 = chance)"],
  ["train_accuracy", "the same model scored on its own 27 training people, averaged over folds"],
  ["overfit_gap", "train minus test: large = the model memorised its training people (only a warning, never the result)"],
], { colW: [3.0, 9.1], rowH: 0.48, fs: 14,
  qa: ["Why is the confidence interval so wide?", "With 28 people, one person = 3.6 points. The data cannot pin the accuracy down more precisely."] });

tf("How AUC and the 95% confidence interval are computed", [
  "**AUC** (area under the ROC curve) = the chance that a randomly chosen patient gets a **higher** P(patient) than a randomly chosen control",
  "- with 14 + 14 people there are 14 x 14 = 196 patient-control pairs; count the pairs ordered correctly (ties count 1/2) and divide by 196",
  "- 1.0 = every patient above every control; 0.5 = random order; 0.24 (the LSTM) = mostly the wrong way round",
  "- AUC uses the probabilities, not the 0.5 cut-off: it can be good even when accuracy is not, and the other way round",
  "**95% CI** (Wilson method), with p = accuracy, n = number of people, z = 1.96:",
  "- centre = (p + z^2/2n) / (1 + z^2/n); half-width = z x sqrt( p(1-p)/n + z^2/4n^2 ) / (1 + z^2/n)",
  "- example: 23 of 28 correct, p = 0.821 -> **64.4% to 92.1%**",
  "- Wilson instead of 'p +/- 1.96 x SE' because it behaves well with small n and never goes below 0 or above 1",
  "Code: evaluate.wilson_ci; AUC from sklearn.metrics.roc_auc_score",
], null);

tf("Why some accuracies fall below 50%", [
  "With 14 vs 14: hold out a control -> training has **13 controls, 14 patients** -> a weak model leans towards 'patient' -> the held-out control is called a patient",
  "The same happens in reverse for every held-out patient",
  "So a model that learns nothing useful scores **below** 50% under LOSO, not at 50%",
  "Examples: decision tree 50% / 41%, stacking + SVM 0%, LSTM 36%",
  "This is why the permutation test (Module 11) is needed: it measures what chance really gives with this exact setup",
], null, { qa: ["Isn't 36% better than 50% if we flip it?", "No: it means no real signal plus the LOSO imbalance. Flipping would be using the test results to choose."] });

tf("The rules every analysis follows", [
  "1. **Split by person**: all epochs of a person on one side, in every split, bootstrap and permutation",
  "2. **Learn only from the training part**: scaling, feature selection, PCA, tuning and (since November) the **maps** are fitted inside each training fold",
  "3. **LOSO** unless stated; epoch predictions averaged per person; epoch-level accuracy never reported",
  "4. **Report twice**: with s14 (main) and without s14",
  "5. No data augmentation, no overlapping windows",
  "6. Feature names always from features_final_16_list.txt",
], null);


/* =====================================================================================
   MODULE 8 - CLASSICAL ML
   ===================================================================================== */
section(8, "Classical ML on the 16 features", ["features_final_16.csv (28 x 16)"], ["accuracy tables for many classifiers", "feature selection, LIME, clustering results"],
  ["labs/lab09_a3_classifiers.py (item 1)", "baseline_24_vs_16.py, item4_feature_reduction.py", "labs/lab09_a4_pipeline.py, lab09_a5_lime.py (item 5)", "item7_clustering.py, collect_items.py"],
  "Note: these results use the maps fitted once on all 28. Module 11 redoes the best ones with maps refitted in every fold: those are the main results now.");

table("Item 1: 11 classifiers (maps fitted once on all 28)", [
  ["Classifier", "With s14", "Without s14", "Train acc. (with)", "Classifier", "With s14", "Without s14", "Train acc. (with)"],
  ["Logistic regression", "67.9%", "59.3%", "88%", "Random forest", "64.3%", "59.3%", "100%"],
  ["SVM linear", "64.3%", "63.0%", "85%", "AdaBoost", "78.6%", "70.4%", "100%"],
  ["SVM RBF", "67.9%", "66.7%", "92%", "XGBoost", "57.1%", "51.9%", "100%"],
  ["Decision tree", "50.0%", "40.7%", "96%", "CatBoost", "60.7%", "59.3%", "100%"],
  ["Naive Bayes", "60.7%", "55.6%", "73%", "MLP", "50.0%", "51.9%", "100%"],
  ["kNN (k = 5)", "67.9%", "66.7%", "79%", "", "", "", ""],
], { colW: [2.0, 1.1, 1.25, 1.55, 2.0, 1.1, 1.25, 1.55], rowH: 0.42, fs: 13,
  note: "Many models reach 100% on their training people but 50-65% on the test person: **overfitting** with 27 training rows and 16 features. Simple models (logistic regression, kNN, SVM) overfit less.", noteY: 4.6,
  qa: ["Which is best?", "AdaBoost 78.6% here, but differences of 1-3 people are within noise. The pre-chosen candidates were tested properly in Module 11."] });

table("The settings of the 11 classifiers (labs/lab09_a3_classifiers.py)", [
  ["Classifier", "Settings (scikit-learn unless stated)", "In one line"],
  ["Logistic regression", "LogisticRegression(max_iter=5000), default C = 1, L2 penalty", "weighted sum of features -> probability"],
  ["SVM (linear)", "SVC(kernel='linear'), default C = 1, wrapped in CalibratedClassifierCV for probabilities", "best straight boundary with the widest margin"],
  ["SVM (RBF)", "SVC(kernel='rbf'), default C = 1, gamma = 'scale', calibrated the same way", "curved boundary"],
  ["Decision tree", "DecisionTreeClassifier(max_depth=3)", "up to 3 yes/no questions"],
  ["Random forest", "RandomForestClassifier(n_estimators=300)", "300 trees on random subsets, vote"],
  ["AdaBoost", "AdaBoostClassifier(n_estimators=100)", "100 small trees, each fixing the last one's mistakes"],
  ["XGBoost", "XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.1)", "gradient-boosted shallow trees"],
  ["CatBoost", "CatBoostClassifier(iterations=300, depth=3)", "gradient boosting, another library"],
  ["Naive Bayes", "GaussianNB()", "Bayes' rule with a bell curve per feature and group"],
  ["kNN", "KNeighborsClassifier(n_neighbors=5)", "vote of the 5 most similar people"],
  ["MLP", "MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000)", "small neural network, 10 hidden units"],
  ["L1 + logistic regression", "SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.5)) -> LogisticRegression", "keep features with non-zero L1 weight, then fit"],
], { colW: [2.4, 6.2, 3.5], rowH: 0.38, fs: 11,
  note: "All use random_state = 42 and a StandardScaler fitted inside each training fold. Settings were fixed in advance (not tuned on the test people).", noteY: 6.45 });

table("Baseline check: the same 11 classifiers on all 24, then on the 16", [
  ["Classifier", "24, with s14", "16, with s14", "24, without s14", "16, without s14"],
  ["Logistic regression", "64.3%", "67.9%", "59.3%", "59.3%"],
  ["SVM (linear)", "60.7%", "64.3%", "59.3%", "63.0%"],
  ["SVM (RBF)", "60.7%", "67.9%", "66.7%", "66.7%"],
  ["Decision tree", "60.7%", "50.0%", "51.9%", "40.7%"],
  ["Random forest", "60.7%", "64.3%", "51.9%", "59.3%"],
  ["AdaBoost", "64.3%", "78.6%", "55.6%", "70.4%"],
  ["XGBoost", "50.0%", "57.1%", "48.1%", "51.9%"],
  ["CatBoost", "60.7%", "60.7%", "59.3%", "59.3%"],
  ["Naive Bayes", "60.7%", "60.7%", "59.3%", "55.6%"],
  ["kNN (k = 5)", "71.4%", "67.9%", "70.4%", "66.7%"],
  ["MLP", "57.1%", "50.0%", "55.6%", "51.9%"],
], { colW: [3.3, 2.2, 2.2, 2.2, 2.2], rowH: 0.33, fs: 13,
  note: "scripts/baseline_24_vs_16.py. With all 24, most models reach 57-64% and kNN is best. Removing the redundant columns helps most models, especially AdaBoost (+14 points), but not all (kNN and the tree drop). Most differences are 1-3 people: call them small.", noteY: 5.55 });

table("Item 4: feature reduction (selection always inside each fold)", [
  ["Feature set", "Logistic reg. with", "without", "AdaBoost with", "without"],
  ["all 24", "64.3%", "59.3%", "64.3%", "55.6%"],
  ["16 (structure rule, no labels)", "67.9%", "59.3%", "78.6%", "70.4%"],
  ["top-4 of 16 (ANOVA F, in fold)", "71.4%", "70.4%", "67.9%", "63.0%"],
  ["top-8 of 16", "67.9%", "66.7%", "75.0%", "70.4%"],
  ["top-12 of 16", "64.3%", "63.0%", "78.6%", "77.8%"],
  ["L1 selection of 16 (in fold)", "78.6%", "81.5%", "67.9%", "66.7%"],
], { colW: [4.1, 2.2, 1.8, 2.2, 1.8], rowH: 0.42, fs: 14,
  note: "Fewer features helps the linear model: **L1 selection + logistic regression** was the best (78.6% / 81.5%) and became a pre-chosen candidate for the permutation test.", noteY: 4.7 });

table("Item 4: which features 'top-4 by ANOVA' picked in the 28 folds", [
  ["Feature", "Folds selected (of 28)", "Feature", "Folds selected (of 28)"],
  ["D occurrence", "28", "B duration", "3"],
  ["C duration", "28", "B -> A", "1"],
  ["A -> B", "28", "A duration", "1"],
  ["A -> D", "23", "all other 9 features", "0"],
], { colW: [3.0, 3.05, 3.0, 3.05], rowH: 0.45, fs: 15,
  note: "From outputs/items/item4_top4_selection_counts.csv (main maps, with s14). The choice is made inside each training fold, yet the same features win almost every time: C duration, A -> B, D occurrence and A -> D. These are also the features with the smallest rank-sum p-values, and L1 keeps C duration and A -> B in every fold of the main model.", noteY: 3.9 });

tf("L1 selection + logistic regression: how it works", [
  "Inside each fold, on the 27 training people only:",
  "1. Scale the 16 features (StandardScaler)",
  "2. Fit a logistic regression with an **L1 penalty** (C = 0.5, liblinear): L1 pushes weak weights to **exactly 0**",
  "3. Keep only the features with a non-zero weight (SelectFromModel)",
  "4. Fit an ordinary logistic regression on the kept features",
  "5. Predict the held-out person",
  "Different folds may keep slightly different features: that is fine, the choice never sees the test person",
], null, { qa: ["Why does fewer features help?", "With 27 people, 16 weights are too many to estimate well. Removing weak features reduces overfitting."] });

tf("Item 5: pipeline and LIME (why did the model decide this?)", [
  "Pipeline = scale -> top-k (ANOVA) -> classifier, all fitted inside the fold. Top-4: 71.4% / 70.4%",
  "**LIME** explains one prediction:",
  "- make many slightly changed copies of that person's features",
  "- ask the model for each copy",
  "- fit a small linear model to those answers: its weights show which features pushed towards 'patient' or 'control'",
  "Shown: logistic regression, with s14, summarised over the held-out people",
  "LIME explains the **model**, not the brain",
], path.join(P, "items", "item5_lime_logistic_with_s14.png"), { textW: 5.6 });

table("PCA and stacking (Lab 10)", [
  ["Experiment", "Result with s14", "Comment"],
  ["Logistic regression, original 16 vs PCA 99%", "67.9% vs 67.9%", "PCA fitted inside each fold; no gain"],
  ["AdaBoost, original 16 vs PCA 99%", "78.6% vs 64.3%", "PCA hurts the tree-based model"],
  ["Random forest, original vs PCA", "64.3% vs 67.9%", "within noise"],
  ["Soft voting (average of base models)", "67.9%", "reference for stacking"],
  ["Stacking + logistic regression", "32.1%", "below chance: nested LOSO + tiny data"],
  ["Stacking + SVM", "0.0%", "extreme form of the below-50% effect"],
], { colW: [5.0, 2.6, 4.5], rowH: 0.45, fs: 14,
  note: "Stacking adds another learning layer on top of tiny data, so it overfits and shows the LOSO imbalance effect most strongly.", noteY: 4.9 });

tf("Item 7: clustering never sees the labels", [
  "k-means, Ward hierarchical and Gaussian mixture on the 16 standardised features, k = 2, 3, 4",
  "Compare clusters with the diagnosis:",
  "- **ARI** (0 = chance, 1 = identical): all between -0.03 and 0.07",
  "- **purity**: 0.59-0.70 (0.5 = chance with 14 vs 14)",
  "- permutation p: all > 0.09",
  "Conclusion: the natural groups in the features are **not** the diagnosis groups; supervised learning is needed",
], path.join(P, "items", "item7_kmeans_pca.png"), { textW: 5.6 });


/* =====================================================================================
   MODULE 9 - LEAKAGE FIX + PERMUTATION
   ===================================================================================== */
section(9, "Maps inside every fold + the chance check", ["cached GFP peaks of all 28", "epochs_15min.npz", "labels"], ["fold maps (55 refits)", "12 classifiers x 2 map versions", "permutation p-values"],
  ["nested_loso_refit_maps.py", "permutation_test.py (old version)", "permutation_test_nested.py"],
  "This is the faculty's point: 'all the preprocessing inside the cross-validation'. These are now the main classification results.");

tf("The small leak, and why we fixed it", [
  "Before: the 4 maps were fitted **once on all 28 people**, then LOSO on the features",
  "No labels were used, but the **test person's EEG helped make the maps** that later describe them",
  "This is a mild, unsupervised leak; the faculty asked for **everything inside the cross-validation**",
  "Fix: in the fold where person t is tested, fit the maps **without t**",
  "Cost: 28 refits (27 without s14) of modified k-means, 50 starts each: about 1 hour, run in parallel (8 workers)",
], null, { qa: ["Was the old result cheating?", "Not with labels, but the test person was not fully unseen. Refitting removes that doubt; the maps turned out nearly identical."] });

flow("One fold of the nested LOSO (test person = t)", [
  ["1. Peaks", "the same 1,500 peaks per person (cached), all except t"],
  ["2. Fit", "modified k-means, k = 4, 50 starts, seed 42 -> fold maps"],
  ["3. Match", "Hungarian matching to the main maps by |r|: fixes column order only"],
  ["4. Label", "label every kept epoch of all people with the fold maps"],
  ["5. Features", "24 -> 16 with the same rule"],
  ["6. Train", "scaler + model on the 27 others"],
  ["7. Test", "predict t"],
  ["8. Repeat", "for all 28 (and 27 without s14)", true],
], { perRow: 4, bh: 1.45, rgap: 0.6,
  note: "Labelling uses the saved epochs (epochs_15min.npz), about 1 s per person. Check: refitting on all 28 reproduces the saved model exactly.",
  qa: ["Why match the maps to the main maps?", "k-means numbers clusters randomly. Matching keeps 'state0' = A in every fold, so the feature columns mean the same thing. No labels used."] });

tf("The train/test split, fold 1 worked through (test person = h01)", [
  "**Test**: h01 (control). **Train**: the other 27 = 13 controls + 14 patients",
  "1. Maps: modified k-means on the 27 x 1,500 = **40,500** peak maps (h01's peaks not used) -> 4 fold maps (|r| >= 0.996 with the main maps)",
  "2. Features: every person's epochs labelled with the fold maps -> 28 x 16 table for this fold",
  "3. StandardScaler: mean and SD of each feature from the **27 training people only**",
  "4. L1 logistic regression on the 27 keeps **C duration and A -> B**; ordinary logistic regression is fitted on those 2",
  "5. Training accuracy: **21 of 27** training people right = 77.8%",
  "6. Test: h01 gets P(patient) = **0.186** -> predicted control -> **correct**",
  "Repeat for all 28 people (next slides): **23 of 28 correct = 82.1%**",
], null, { qa: ["What exactly is 'train' and 'test' here?", "Train = 27 people (everything is learned from them, even the maps); test = 1 person, used only once, at the end of the fold."] });

foldTables("with s14");

tf("Fold by fold, with s14: what the 28 folds show", [
  "**23 of 28 correct = 82.1%** test accuracy; mean training accuracy **85.2%** (gap 3 points: little overfitting)",
  "Wrong: **h08** (0.722) and **h10** (0.612), controls called patients; **s10** (0.426), **s11** (0.257), **s13** (0.334), patients called controls",
  "Close calls near 0.5: h02 0.494, s03 0.501, h06 0.471, h07 0.448: a tiny change can flip them",
  "L1 kept **C duration and A -> B in every fold** (B -> D added in 2 folds): the same two features with the lowest rank-sum p-values",
  "Training set is always 13 + 14: one class is the majority in every fold (the below-50% effect, Module 9)",
], null);

foldTables("without s14");

tf("Fold by fold, without s14: what the 27 folds show", [
  "**18 of 27 correct = 66.7%**; mean training accuracy **82.2%**",
  "Wrong: h02 (0.527), h07 (0.530), h08 (0.775), h10 (0.591); s03 (0.485), s04 (0.424), s10 (0.365), s11 (0.152), s13 (0.279)",
  "Removing one patient (s14) flips **h02, h07, s03, s04** across the 0.5 line: s03 0.501 -> 0.485, h02 0.494 -> 0.527",
  "L1 again keeps C duration and A -> B in every fold, sometimes D duration too",
  "This is the **instability**: the same model, nearly the same data, 4 people move -> 82.1% becomes 66.7%",
], null, { qa: ["Why does one person change the result so much?", "Several people sit close to P = 0.5, and each person is 3.6-3.7 points. Removing s14 changes every training set slightly."] });

tf("The fold maps are almost the same as the main maps", [
  "outputs/nested_loso/fold_map_similarity.csv: 55 refits",
  "Lowest |r| between a fold map and its main map: **0.996**",
  "Fold GEV: 0.668-0.687 (with s14), 0.680-0.700 (without)",
  "Yet the accuracies change by **1-4 people** (next slide)",
  "Meaning: the results are **unstable**: tiny changes in the features move a few people across the 0.5 line",
], null, { qa: ["If the maps are the same, why do results change?", "Several people sit close to P = 0.5. A tiny shift in their features flips them. With 28 people each flip is 3.6 points."] });

table("Main maps vs maps refitted in each fold", [
  ["Classifier", "Main maps with s14", "Refitted with s14", "Main maps without", "Refitted without"],
  ["L1 selection + logistic regression", "78.6%", "82.1%", "81.5%", "66.7%"],
  ["AdaBoost", "78.6%", "71.4%", "70.4%", "59.3%"],
  ["Logistic regression", "67.9%", "67.9%", "59.3%", "63.0%"],
  ["SVM (RBF)", "67.9%", "71.4%", "66.7%", "77.8%"],
  ["kNN (k = 5)", "67.9%", "75.0%", "66.7%", "66.7%"],
  ["MLP", "50.0%", "71.4%", "51.9%", "74.1%"],
  ["Decision tree", "50.0%", "50.0%", "40.7%", "44.4%"],
  ["Naive Bayes", "60.7%", "50.0%", "55.6%", "51.9%"],
], { colW: [3.9, 2.05, 2.05, 2.05, 2.05], rowH: 0.42, fs: 14,
  note: "Only the 3 candidates chosen **before** (L1 + LR, AdaBoost, LR) are tested for chance. SVM/MLP jumps are not used: picking the best of 12 after seeing them would be cherry-picking.", noteY: 5.3 });

tf("Permutation test: what does chance give?", [
  "1. Score the model with the real labels (full LOSO)",
  "2. **Shuffle the labels across people** (seed 42 + i) and redo the whole LOSO; record the balanced accuracy",
  "3. Repeat **1,000** times -> the 'chance' distribution (histogram)",
  "4. p = (1 + number of shuffles >= real) / (1 + 1,000)",
  "**Family-wise p**: for each shuffle take the best of the 3 candidates, so testing 3 models is paid for",
  "Fold maps are label-free, so they need no refit per shuffle",
], path.join(P, "nested_loso", "permutation_nested_with_s14.png"), { stack: true, textH: 2.6,
  caption: "Bars = chance distribution from 1,000 shuffles; line = the real score (with s14)." });

table("Permutation results (maps refitted in each fold): the main results", [
  ["Candidate", "With s14: acc.", "chance mean", "p", "p family", "Without s14: acc.", "p", "p family"],
  ["L1 selection + logistic regression", "82.1%", "44.7%", "0.004", "0.011", "66.5%*", "0.089", "0.153"],
  ["AdaBoost", "71.4%", "47.7%", "0.041", "0.073", "59.3%*", "0.171", "0.324"],
  ["Logistic regression", "67.9%", "45.5%", "0.047", "0.124", "62.9%*", "0.105", "0.232"],
], { colW: [3.3, 1.35, 1.25, 0.9, 1.05, 1.65, 1.1, 1.5], rowH: 0.5, fs: 13,
  intro: "From outputs/nested_loso/permutation_nested_summary.csv (balanced accuracy):",
  note: "* Without s14 the table gives balanced accuracy (27 people, 14 vs 13). **Only L1 + LR with s14 beats chance after correction (p = 0.011).** Without s14 nothing does. Older version (maps once on all 28): L1 + LR 78.6% (p = 0.033) and 81.5% (p = 0.014).", noteY: 4.15,
  qa: ["So does it work?", "Partly and fragile: one model beats chance with s14 only. We call it exploratory evidence, not a biomarker."] });

table("Permutation test before the refit (maps fitted once on all 28)", [
  ["Candidate", "With s14: acc.", "chance mean", "p", "p family", "Without s14: bal. acc.", "p", "p family"],
  ["L1 selection + logistic regression", "78.6%", "44.7%", "0.009", "0.033", "81.0%", "0.003", "0.014"],
  ["AdaBoost", "78.6%", "48.0%", "0.021", "0.033", "70.1%", "0.061", "0.120"],
  ["Logistic regression", "67.9%", "46.0%", "0.061", "0.177", "59.1%", "0.194", "0.478"],
  ["LaBraM embeddings + logistic regression", "75.0%", "45.7%", "0.011", "0.059", "74.2%", "0.014", "0.050"],
], { colW: [3.6, 1.4, 1.3, 0.9, 1.0, 1.8, 0.9, 1.2], rowH: 0.5, fs: 13,
  intro: "From outputs/permutation/permutation_summary.csv: 1,000 shuffles, family-wise over these 4 candidates:",
  note: "Before the refit, L1 + LR beat chance both with and without s14, and AdaBoost with s14. After the refit (previous slide) only L1 + LR with s14 does. Same models, nearly identical maps: this is the instability again, and why the refitted (stricter) numbers are the ones we report.", noteY: 4.75 });


/* =====================================================================================
   MODULE 10 - DEEP LEARNING
   ===================================================================================== */
section(10, "Deep learning", ["epochs_15min_labram.npz (raw epochs)", "the microstate sequences (for the LSTM)"], ["LaBraM embeddings + classifier results", "fine-tuning results", "LSTM results"],
  ["make_epoch_array.py --preset labram", "item2_embeddings.py, item3_finetune.py", "item8_lstm_sequences.py"]);

tf("LaBraM: a pretrained EEG model", [
  "LaBraM (Jiang et al., ICLR 2024): a transformer trained on about 2,500 hours of other people's EEG",
  "Weights from Hugging Face 'braindecode/labram-pretrained' (23 MB), downloaded automatically",
  "Input: our epochs, prepared like its training data: **0.1-75 Hz, 50 Hz notch, 200 Hz**",
  "Output: **200 numbers per epoch** (an embedding): a learned summary of the signal",
  "No microstates involved: it is a different route from the raw EEG",
], null, { qa: ["Why a pretrained model?", "28 people are far too few to train a deep network from scratch. A model trained on lots of other EEG can be reused."] });

tf("Inside LaBraM: from an epoch to 200 numbers", [
  "1. Our 5-s epoch, 19 channels, at 200 Hz = 19 x 1,000 samples",
  "2. Each channel is cut into **1-second patches** (200 samples): 19 x 5 = **95 patches**",
  "3. A small convolutional encoder turns every patch into a **token** (a vector of 200 numbers)",
  "4. Two learned 'labels' are added to each token: **which electrode** (spatial embedding) and **which second** (temporal embedding)",
  "5. A **transformer** with 12 layers lets every token look at every other token (attention), across channels and time",
  "6. Our embedding = the **average of the 95 output tokens** = 200 numbers per epoch",
  "Pretraining (done by the authors, not us): parts of the EEG were hidden and the model learned to predict them ('masked EEG modelling'), on about 2,500 hours of EEG",
  "Item 2 uses steps 1-6 frozen; item 3 retrains the last 2 of the 12 layers + a classification head",
], null, { qa: ["What do the 200 numbers mean?", "Nothing a person can name: they are a learned summary of the epoch. We only check whether a classifier can use them."] });

table("Item 2: frozen embeddings + classifiers (LOSO)", [
  ["How", "Classifier", "With s14", "Without s14", "AUC (with)"],
  ["epoch level (train on all epochs, average per person)", "Logistic regression", "75.0%", "74.1%", "0.89"],
  ["epoch level", "Random forest", "71.4%", "74.1%", "0.74"],
  ["subject mean (1 row per person)", "Logistic regression", "75.0%", "74.1%", "0.80"],
  ["subject mean", "SVM linear", "75.0%", "74.1%", "0.75"],
  ["subject mean", "kNN (k = 5)", "67.9%", "70.4%", "0.67"],
], { colW: [4.8, 2.4, 1.6, 1.6, 1.7], rowH: 0.45, fs: 14,
  note: "Frozen = LaBraM is not changed. Permutation test (shared protocol): p = 0.059 with s14, 0.050 without: borderline, not significant after correction.", noteY: 4.4 });

tf("Item 3: fine-tuning LaBraM (transfer learning)", [
  "The last **2 of its 12** transformer layers + a small classification head are trained on our epochs; the rest stays frozen",
  "Protocol: **7-fold subject-grouped** (StratifiedGroupKFold, about 4 people per test fold), not LOSO: 7 trainings instead of 28 (about 1 hour)",
  "Result: **71.4%** with s14, **77.8%** without; AUC 0.82 / 0.78",
  "Training accuracy **100%** in every fold: strong overfitting",
  "Not permutation-tested; different protocol, so not directly comparable with the LOSO numbers",
], null);

tf("Item 8: LSTM on the microstate sequence", [
  "Idea (faculty: 'the sequential thing'): features summarise one step; an LSTM reads the **whole sequence** of an epoch",
  "Input per epoch: its segments in order, e.g. A (64 ms) -> D (112 ms) -> B (48 ms) ...; each step = one-hot state + log duration (scaled on training data)",
  "Model: LSTM 32 units -> dropout 0.3 -> linear; Adam, lr 1e-3, weight decay 1e-4, batch 64, 20 passes",
  "Person's prediction = mean of its epoch probabilities; maps refitted in each fold (Module 11)",
  "**Control**: the same with the segment order **shuffled** inside each epoch",
], null);

table("LSTM result: no signal (goes under limitations and future work)", [
  ["Input", "With s14", "Without s14", "AUC with / without"],
  ["real order", "35.7%", "48.1%", "0.24 / 0.38"],
  ["shuffled order (control)", "35.7%", "29.6%", "0.27 / 0.20"],
], { colW: [4.0, 2.4, 2.4, 3.3], rowH: 0.55, fs: 15,
  note: "At or below chance (see the below-50% effect, Module 9), and no better than shuffled order: the LSTM found no usable order information. Likely reasons: 28 people are too few for a sequence model; the sequences are long and noisy. **Faculty: report it as a limitation and future work, not highlighted.**", noteY: 3.2 });

table("LSTM per person (with s14, real order): why 35.7%", [
  ["Control", "P(patient)", "Predicted", "Patient", "P(patient)", "Predicted"],
  ["h01", "0.52", "SC (wrong)", "s01", "0.53", "SC"],
  ["h02", "0.50", "SC (wrong)", "s02", "0.48", "HC (wrong)"],
  ["h03", "0.54", "SC (wrong)", "s03", "0.31", "HC (wrong)"],
  ["h04", "0.62", "SC (wrong)", "s04", "0.51", "SC"],
  ["h05", "0.58", "SC (wrong)", "s05", "0.53", "SC"],
  ["h06", "0.54", "SC (wrong)", "s06", "0.54", "SC"],
  ["h07", "0.60", "SC (wrong)", "s07", "0.53", "SC"],
  ["h08", "0.64", "SC (wrong)", "s08", "0.51", "SC"],
  ["h09", "0.57", "SC (wrong)", "s09", "0.48", "HC (wrong)"],
  ["h10", "0.55", "SC (wrong)", "s10", "0.56", "SC"],
  ["h11", "0.51", "SC (wrong)", "s11", "0.36", "HC (wrong)"],
  ["h12", "0.58", "SC (wrong)", "s12", "0.58", "SC"],
  ["h13", "0.65", "SC (wrong)", "s13", "0.47", "HC (wrong)"],
  ["h14", "0.47", "HC", "s14", "0.57", "SC"],
], { colW: [1.6, 1.6, 2.2, 1.6, 1.6, 2.2], rowH: 0.29, fs: 12,
  note: "Almost every probability is close to 0.5 (most between 0.45 and 0.65): the LSTM is unsure about everybody. It called 13 of 14 controls patients, and controls even got slightly HIGHER P(patient) than patients, so AUC = 0.24 (it ranks backwards). 1 + 9 = 10 of 28 correct = 35.7%. File: outputs/lstm/lstm_per_subject.csv", noteY: 5.95 });


/* =====================================================================================
   MODULE 11 - COMPARISON
   ===================================================================================== */
section(11, "Comparison with published work", ["our result files", "published papers on the same dataset"], ["item6_comparison.csv / .md"],
  ["item6_comparison.py"]);

table("Same dataset, different validation", [
  ["Study", "Validation", "Accuracy", "Comparable?"],
  ["Kim et al. 2021, PLOS ONE [9]", "10-fold, subject-wise; permutation test", "75.6%", "yes: same preprocessing, subject-wise"],
  ["Keihani et al. 2022, Brain Sciences [10]", "random 80/20 split of EPOCHS", "90.9%", "no: same people in train and test"],
  ["Raeisi et al. 2025, Cogn. Neurodynamics", "data augmented from 14 to 56 per group", "99.9%", "no: augmentation"],
  ["Racz & Csukly 2025 (preprint)", "re-ran pipelines with and without leakage", "gap 28 points on this dataset", "explains the high numbers"],
  ["Ours: microstates, maps refitted per fold", "LOSO + permutation (3 candidates)", "82.1% / 66.7%", "p = 0.011 / 0.153"],
  ["Ours: frozen LaBraM", "LOSO + permutation", "75.0% / 74.1%", "p = 0.059 / 0.050"],
], { colW: [3.6, 3.6, 2.2, 2.7], rowH: 0.5, fs: 13,
  note: "Our protocol is the strictest one: subject-wise, maps inside the folds, chance measured by permutation. Our numbers should be compared with Kim et al. (75.6%), not with 90-99%.", noteY: 5.1 });

table("Step by step: ours vs Keihani et al. 2022 (same dataset)", [
  ["Step", "Ours", "Keihani et al. 2022 [10]"],
  ["Software", "Python: MNE + pycrostates", "MATLAB: EEGLAB"],
  ["Filter, reference", "2-20 Hz, average reference", "2-20 Hz, common average (same)"],
  ["Epochs, bad epochs", "5 s, first 15 min; robust z of log-variance > 3.5", "5 s; variance > 3 SD (similar)"],
  ["Maps", "modified k-means, k = 4, 50 starts, one shared label-free model", "modified k-means, k = 4, 20 restarts; separate maps per group in their figure"],
  ["GEV", "67.4%", "about 70%"],
  ["Labelling", "every sample (backfitting) + smoothing, edges rejected", "each GFP peak labelled"],
  ["Features", "duration, occurrence, coverage + 12 transitions = 24 (16 free)", "duration, occurrence, coverage + mean GFP + map correlation = 20; no transitions"],
  ["Feature ranking", "inside each training fold", "chi-square on all the data, before splitting"],
  ["Validation", "leave-one-subject-out", "random 80/20 split of epochs"],
  ["Group statistics", "per person (n = 28), rank-sum", "t-tests on about 2,500 epochs per group (t up to 46.9)"],
], { colW: [2.4, 4.9, 4.8], rowH: 0.42, fs: 12 });

tf("Why our accuracy is lower, and why it is the honest number", [
  "**1. Same people in train and test.** A random split of epochs puts about 170 epochs of the same person on both sides: the model can recognise the person. Racz & Csukly (2025) found this adds about 28 points on this dataset",
  "**2. Feature ranking on all data.** Ranking before splitting lets the test data help choose the features; we choose inside each training fold",
  "**3. Statistics on epochs.** t-tests on 2,500 epochs treat epochs as independent people, so p-values become tiny for small effects; we test per person (n = 28)",
  "**Kept from Keihani et al.**: the preprocessing, modified k-means with k = 4, the feature definitions",
  "**Added**: transitions, one label-free shared model (refitted per fold), subject-wise validation, confidence intervals, permutation test, the s14 check",
  "Kim et al. (subject-wise, 10-fold): 75.6%. Our 75-82% for the best models is in the same range",
], null);


/* =====================================================================================
   MODULE 12 - LABS
   ===================================================================================== */
section(12, "Lab exercises on the project data", ["features_final_16.csv"], ["outputs/labs/ (tables and figures)"],
  ["labs/labNN_aX_*.py, labs/lab_common.py", "Lab 8 A1-A11 are separate submissions, not here"]);

table("Labs 3-6", [
  ["Lab", "Files", "What it does on our data", "Result"],
  ["3", "lab03_functions.py, lab03_a11_project.py", "own functions: mean, variance, Minkowski distance, histogram, own k-means", "k-means clusters vs diagnosis: 8 HC + 3 SC / 6 HC + 11 SC"],
  ["4", "lab04_functions.py, lab04_a1_a3_project.py, lab04_a2_unit_tests.py", "AI-tool versions of Lab 3 functions, unit tests, comparison", "ARI vs diagnosis: own 0.10, AI and sklearn -0.02"],
  ["5", "lab05_knn.py, lab05_a3_a7_split.py, lab05_a8_a9_k_sweep.py", "own kNN and weighted kNN; 70/30 split; k = 1-15", "changes with k (next slide)"],
  ["6", "lab06_ai_knn.py, lab06_a3_comparison.py, lab06_a2_unit_tests.py", "own vs sklearn vs AI kNN (k = 3)", "identical predictions; LOSO 64.3% / 63.0%"],
], { colW: [0.6, 3.9, 4.2, 3.4], rowH: 0.75, fs: 12,
  note: "The 70/30 split (Lab 5) tests only 9 people: one person = 11 points. That is why the project uses LOSO.", noteY: 5.4 });

tf("Lab 5: the k sweep (kNN)", [
  "Accuracy for k = 1 ... 15: sklearn kNN, own kNN, own weighted kNN; 70/30 split and LOSO",
  "The curve jumps around: with 28 people one person moves it by 3.6 points (LOSO) or 11 points (70/30)",
], path.join(P, "labs", "lab05_a8_a9_k_sweep.png"), { stack: true, textH: 1.0 });

table("Labs 7-10", [
  ["Lab", "Files", "What it does", "Result (with s14)"],
  ["7", "lab07_tree.py, lab07_a6_a7_plots.py, lab07_a8_gridsearch.py", "own entropy, Gini, binning, root node, decision tree; plots; GridSearchCV nested in LOSO", "own tree 32%, sklearn 43%, tuned 39% (overfits)"],
  ["8", "lab08_a12_mlp.py", "MLP of several sizes, LOSO", "36-57%, train 100%"],
  ["9", "lab09_a2_tuning.py, a3, a4, a5", "tuning in the fold; 11 classifiers (item 1); pipeline; LIME (item 5)", "Perceptron default 75.0%, tuned 53.6%"],
  ["10", "lab10_a1_stacking.py, a2_heatmap.py, a3_pca.py", "stacking; feature correlation heatmap; PCA 99%", "stacking 0-61%; PCA no gain"],
], { colW: [0.6, 3.9, 4.6, 3.0], rowH: 0.8, fs: 12,
  note: "Tuning inside the fold sometimes makes the result worse (Perceptron 75% -> 54%): with 27 training people, tuning mostly fits noise.", noteY: 5.85 });

tf("Lab 7: the decision tree", [
  "sklearn tree (entropy, depth 3) fitted on all 28 for the picture: easy to read, but under LOSO it scores only 39-43% (overfits)",
], path.join(P, "labs", "lab07_a6_tree.png"), { stack: true, textH: 0.6 });

tf("Lab 10 A2: how to read the feature correlation heatmap", [
  "Each cell = **Pearson r** between two features across the 28 people (-1 to +1)",
  "- r = sum (x - mean x)(y - mean y) / sqrt( sum (x - mean x)^2 x sum (y - mean y)^2 )",
  "Dark red near +1 = strong positive; dark blue near -1 = strong negative; white near 0 = no linear relation",
  "Diagonal = 1 (a feature with itself); only the lower half is drawn because r(A,B) = r(B,A)",
  "Examples: A occurrence vs D->A **+0.88**; A occurrence vs D duration **-0.81**; A duration vs A occurrence **+0.02**",
  "Strongly correlated pairs overlap in information: one reason fewer features (L1) helps",
  "No labels used, so using all 28 is fine. Correlation is not causation",
], path.join(P, "labs", "lab10_a2_heatmap.png"), { textW: 5.6 });

table("Three different 'heatmaps' in the project", [
  ["Figure", "What the colours are", "How it is made"],
  ["Scalp maps A-D (maps_labelled_ABCD.png)", "the map's 19 values; red and blue = opposite signs (polarity ignored)", "mne.viz.plot_topomap puts the 19 electrodes at their 10-20 positions on a flat head (nose up) and interpolates between them. plot_labelled_maps.py"],
  ["Transition matrix (transition_matrix_heatmap.png)", "cell (row i, column j) = mean P(i -> j) over a group; third panel = patients minus controls", "average each trans column over controls and over patients, arrange 4 x 4 (diagonal empty), colour with imshow. plot_transition_matrix.py"],
  ["Feature correlations (lab10_a2_heatmap.png)", "Pearson r between two features over 28 people", "df[features].corr() -> 16 x 16, seaborn heatmap, lower triangle. labs/lab10_a2_heatmap.py"],
], { colW: [3.3, 3.9, 4.9], rowH: 0.95, fs: 13 });


/* =====================================================================================
   MODULE 13 - SUMMARY
   ===================================================================================== */
section(13, "Results summary, limitations, future work", ["everything above"], ["what we can say", "what we cannot say", "what comes next"],
  ["README.md (main results table)", "outputs/items/README.md"]);

table("All main results in one table", [
  ["Approach", "With s14", "Without s14", "Better than chance? (corrected)"],
  ["Microstates, maps refitted per fold, L1 + logistic regression (MAIN)", "82.1%", "66.7%", "with s14 yes (p = 0.011); without no (0.153)"],
  ["Same, AdaBoost", "71.4%", "59.3%", "no (0.073 / 0.324)"],
  ["Same, logistic regression", "67.9%", "63.0%", "no (0.124 / 0.232)"],
  ["Older: maps once on all 28, L1 + logistic regression", "78.6%", "81.5%", "yes (0.033 / 0.014)"],
  ["Frozen LaBraM + logistic regression", "75.0%", "74.1%", "borderline (0.059 / 0.050)"],
  ["Fine-tuned LaBraM (7-fold)", "71.4%", "77.8%", "not tested"],
  ["LSTM on sequences (limitation)", "35.7%", "48.1%", "no"],
  ["Published, subject-wise (Kim et al. 2021)", "75.6%", "", ""],
], { colW: [5.6, 1.4, 1.5, 3.6], rowH: 0.45, fs: 13 });

tf("What we can say, and what we cannot", [
  "**Can say**",
  "- Microstate features can partly separate patients and controls with a strict, leakage-free protocol (82.1%, p = 0.011, with s14)",
  "- The 4 maps are stable (|r| >= 0.996 across folds) and match the classic A-D shapes",
  "- C duration and A -> B tend to be higher in patients (uncorrected)",
  "**Cannot say**",
  "- that any feature is a biomarker or a diagnostic test",
  "- that the result holds without s14 (it does not reach significance)",
  "- that patients 'have more frontal activity' (maps do not show that)",
], null);

tf("Limitations and future work", [
  "**Limitations**",
  "- 28 people: wide confidence intervals; results move by 1-4 people with tiny changes (instability)",
  "- s14 strongly affects the main result",
  "- names C and D are the least certain",
  "- the GFP peaks that fit the maps come from each whole recording (s14: all 36 min, including after its change at 18 min), while the features use the first 15 min. No labels are involved, but it is cleaner to take the peaks from the same 15 min",
  "- the LSTM on the sequences did not learn anything usable (at chance, same as shuffled order)",
  "- no clinical information (medication, age) used",
  "**Future work** (with the faculty, semester break)",
  "- a larger or second dataset to test whether the result replicates",
  "- simpler sequence models (e.g. a Markov-chain classifier) instead of an LSTM",
], null);

table("Faculty question bank (practise these)", [
  ["Question", "Short answer (details: module)"],
  ["What is GFP and what are its axes?", "SD across 19 channels per sample; x = time, y = µV (5)"],
  ["Why GFP peaks?", "map strongest and cleanest there; toy Part 2 |r| vs GFP (5)"],
  ["Which library, which function?", "pycrostates extract_gfp_peaks = np.std + scipy find_peaks, distance 2 (5)"],
  ["Why modified k-means?", "polarity ignored (|r|), update by eigenvector, 50 starts (6)"],
  ["Why 4 maps?", "literature A-D, GEV 67.4%, mild elbow at 4 (6)"],
  ["How do you read a map?", "gradient orientation; polarity irrelevant; no 'more activity' (6)"],
  ["What are the features?", "duration, occurrence, transitions: 16 after removing redundancy (7)"],
  ["How did you split train/test?", "LOSO, by person, maps refitted in each fold (9, 11)"],
  ["Is 82% real?", "p = 0.011 corrected with s14; not without s14; exploratory (11)"],
  ["Why do papers get 99%?", "epoch splits or augmentation leak the person (13)"],
], { colW: [4.6, 7.5], rowH: 0.43, fs: 13 });


pres.writeFile({ fileName: OUT }).then(f => console.log("saved", f));
