# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Self-contained HTML generator for the review annotation interface."""

import html
import json
from typing import List

from patient_agent_bench.review.models import (
    ANNOTATION_DIMENSIONS,
    RUBRIC_DIMENSIONS,
    RUBRIC_GUIDE,
    SampledConversation,
)


class HTMLGenerator:
    """Generates a single self-contained HTML file with all CSS and JS inlined."""

    def generate(
        self,
        conversations: List[SampledConversation],
        annotation_json_filename: str,
    ) -> str:
        """Return complete HTML string with inlined CSS/JS."""
        conv_data = json.dumps(
            [c.model_dump() for c in conversations],
            ensure_ascii=False,
        )
        dims_json = json.dumps(ANNOTATION_DIMENSIONS)
        rubric_dims_json = json.dumps(RUBRIC_DIMENSIONS)

        nav_items = self._build_nav_items(conversations)
        escaped_filename = html.escape(annotation_json_filename)
        n = len(conversations)

        return self._render_html(
            conv_data, dims_json, rubric_dims_json,
            nav_items, escaped_filename, n,
        )

    def _build_nav_items(
        self, conversations: List[SampledConversation],
    ) -> str:
        items = []
        for i, conv in enumerate(conversations):
            case_id = html.escape(conv.case_id)
            exp_id = html.escape(conv.experiment.experiment_id)
            label = html.escape(conv.experiment.assistant_label)
            items.append(
                f'<div class="nav-item" data-index="{i}" '
                f'onclick="showConversation({i})">'
                f'<span class="nav-check" id="nav-check-{i}">'
                f"</span>"
                f'<div class="nav-text">'
                f'<div class="nav-case">{i + 1}. {case_id}</div>'
                f'<div class="nav-exp">'
                f'{exp_id} &middot; {label}</div>'
                f"</div></div>"
            )
        return "\n".join(items)


    def _render_html(
        self,
        conv_data: str,
        dims_json: str,
        rubric_dims_json: str,
        nav_items: str,
        annotation_json_filename: str,
        num_conversations: int,
    ) -> str:
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" "
            "content=\"width=device-width, initial-scale=1.0\">\n"
            "<title>PatientAgentBench - Review &amp; "
            "Annotation</title>\n"
            f"<style>\n{self._css()}\n</style>\n"
            "</head>\n<body class=\"dark\">\n"
            "<div id=\"app\">\n"
            "  <nav id=\"sidebar\">\n"
            "    <div id=\"sidebar-header\">\n"
            "      <h1>PatientAgentBench</h1>\n"
            f"      <h2>Conversations ({num_conversations})</h2>\n"
            "    </div>\n"
            f"    {nav_items}\n"
            "  </nav>\n"
            "  <main id=\"main-content\">\n"
            "    <div id=\"top-bar\">\n"
            "      <div id=\"header-bar\">\n"
            "        <span id=\"conv-title\"></span>\n"
            "        <span style=\"display:flex;align-items:center;"
            "gap:8px;\">"
            "<span id=\"annotation-warn\" "
            "style=\"font-size:11px;color:#f87171;display:none;\">"
            "&#9888; Annotators: stay in annotation mode"
            "</span>"
            "<button id=\"annotation-toggle\" "
            "onclick=\"promptExitAnnotation()\" "
            "style=\"display:none\">"
            "Exit Annotation Mode</button>"
            "<button id=\"theme-toggle\" "
            "onclick=\"toggleTheme()\" "
            "title=\"Toggle light/dark mode\">"
            "&#9788;</button>"
            "<button onclick=\"downloadAnnotations()\" "
            "class=\"download-btn\" "
            f"data-filename=\"{annotation_json_filename}\">"
            "&#11015; Download</button>"
            "</span>\n"
            "      </div>\n"
            "    </div>\n"
            "    <div id=\"three-col\">\n"
            "      <div id=\"context-col\">\n"
            "        <div id=\"scenario-panel\">\n"
            "          <div class=\"panel-header\">"
            "<span>Patient Scenario</span></div>\n"
            "          <div id=\"scenario-content\"></div>\n"
            "        </div>\n"
            "        <div id=\"patient-profile-panel\">\n"
            "          <div class=\"panel-header\">"
            "<span>Patient Profile</span></div>\n"
            "          <div id=\"patient-profile-content\">"
            "</div>\n"
            "        </div>\n"
            "      </div>\n"
            "      <div id=\"conversation-col\">\n"
            "        <div id=\"conversation-messages\"></div>\n"
            "      </div>\n"
            "      <div id=\"scores-col\">\n"
            "        <div id=\"llm-scores-panel\" "
            "class=\"hidden\">\n"
            "          <h3>LLM Evaluator Scores</h3>\n"
            "          <div id=\"llm-scores-content\"></div>\n"
            "        </div>\n"
            "        <div id=\"annotation-panel\">\n"
            "          <h3>Human Annotation</h3>\n"
            "          <div class=\"annotation-field\">\n"
            "            <label for=\"annotator-id\">"
            "Annotator ID</label>\n"
            "            <input type=\"text\" "
            "id=\"annotator-id\" placeholder=\"Your ID\">\n"
            "          </div>\n"
            "          <div id=\"annotation-scores\"></div>\n"
            "          <button id=\"submit-annotation\" "
            "onclick=\"submitAnnotation()\">"
            "Submit Annotation</button>\n"
            "          <div id=\"annotation-status\"></div>\n"
            "        </div>\n"
            "      </div>\n"
            "    </div>\n"
            "  </main>\n"
            "</div>\n"
            f"<script>\n{self._js(conv_data, dims_json, rubric_dims_json)}\n"
            "</script>\n</body>\n</html>"
        )

    def _css(self) -> str:
        return (
            self._css_base()
            + self._css2()
            + self._css3()
            + self._css4()
        )


    def _css_base(self) -> str:
        return """
:root {
  --bg: #1a1a2e; --bg2: #16213e; --card: #1f2937;
  --card-border: #374151; --text: #e2e8f0; --text2: #94a3b8;
  --text3: #64748b; --accent: #3b82f6; --sidebar: #0f172a;
  --msg-ai: #1e3a5f; --msg-ai-border: #2563eb;
  --msg-human: #1f2937; --msg-human-border: #4b5563;
  --msg-tool: #2d2305; --msg-tool-border: #a16207;
  --input-bg: #111827; --input-border: #374151;
}
body.light {
  --bg: #f0f2f5; --bg2: #e5e7eb; --card: #ffffff;
  --card-border: #e5e7eb; --text: #1e293b; --text2: #475569;
  --text3: #64748b; --accent: #3b82f6; --sidebar: #1e293b;
  --msg-ai: #dbeafe; --msg-ai-border: #93c5fd;
  --msg-human: #f1f5f9; --msg-human-border: #cbd5e1;
  --msg-tool: #fef3c7; --msg-tool-border: #fcd34d;
  --input-bg: #ffffff; --input-border: #cbd5e1;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont,
    'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--text);
}
#app { display: flex; height: 100vh; }
#sidebar {
  width: 260px; background: var(--sidebar); color: #e2e8f0;
  overflow-y: auto; padding: 16px; flex-shrink: 0;
}
#sidebar-header { margin-bottom: 12px; }
#sidebar h1 {
  font-size: 15px; color: #60a5fa; margin-bottom: 6px;
  letter-spacing: 0.3px;
}
#sidebar h2 {
  font-size: 12px; color: #94a3b8;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.nav-item {
  padding: 8px 10px; margin-bottom: 4px;
  border-radius: 6px; cursor: pointer;
  transition: background 0.15s;
  display: flex; align-items: center; gap: 8px;
}
.nav-item:hover { background: #334155; }
.nav-item.active { background: var(--accent); color: #fff; }
.nav-check {
  width: 16px; height: 16px; flex-shrink: 0;
  font-size: 13px; line-height: 16px; text-align: center;
}
.nav-check.done { color: #22c55e; }
.nav-check.draft { color: #f59e0b; }
.nav-text { min-width: 0; flex: 1; }
.nav-case {
  font-size: 13px; font-weight: 600;
  white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis;
}
.nav-exp {
  font-size: 11px; color: #94a3b8; margin-top: 2px;
}
.nav-item.active .nav-exp { color: #bfdbfe; }
#main-content {
  flex: 1; display: flex; flex-direction: column;
  padding: 12px 16px; background: var(--bg);
  overflow: hidden; height: 100vh;
}
"""

    def _css2(self) -> str:
        return """
    #top-bar {
      padding: 0 0 10px 0; flex-shrink: 0;
    }
    #header-bar {
      display: flex; justify-content: space-between;
      align-items: center;
    }
    #conv-title { font-size: 18px; font-weight: 600; }
    #annotation-toggle {
      padding: 8px 16px; border: none; border-radius: 6px;
      background: #ef4444; color: #fff; cursor: pointer;
      font-size: 13px;
    }
    #annotation-toggle.inactive { background: var(--accent); }
    .download-btn {
      padding: 7px 14px; background: #6366f1;
      color: #fff; border: none; border-radius: 6px;
      cursor: pointer; font-size: 13px;
    }
    .download-btn:hover { background: #4f46e5; }
    #theme-toggle {
      padding: 6px 10px; background: transparent;
      border: 1px solid var(--card-border); border-radius: 6px;
      color: var(--text2); cursor: pointer; font-size: 16px;
    }
    #theme-toggle:hover { background: var(--card-border); }
    #three-col {
      display: flex; gap: 12px; flex: 1;
      min-height: 0;
    }
    #context-col {
      width: 375px; flex-shrink: 0;
      overflow-y: auto; display: flex;
      flex-direction: column; gap: 12px;
    }
    #scenario-panel {
      background: var(--card); border-radius: 8px;
      border: 1px solid var(--card-border);
      flex-shrink: 0;
    }
    #scenario-content {
      padding: 12px 16px; font-size: 13px;
      line-height: 1.6; color: var(--text2);
    }
    #patient-profile-panel {
      background: var(--card); border-radius: 8px;
      border: 1px solid var(--card-border);
      flex: 1; min-height: 0; display: flex;
      flex-direction: column;
    }
    .panel-header {
      padding: 10px 16px;
      display: flex; justify-content: space-between;
      font-weight: 600; font-size: 14px;
      flex-shrink: 0;
    }
    #patient-profile-content {
      padding: 0 16px 12px; font-size: 12px;
      overflow-y: auto; flex: 1;
    }
    .xml-tag { color: #a78bfa; }
    .xml-attr { color: #60a5fa; }
    .xml-val { color: #34d399; }
    .xml-text { color: var(--text); }
    """

    def _css3(self) -> str:
        return """
    #conversation-col {
      flex: 1; min-width: 0; overflow-y: auto;
    }
    #scores-col {
      width: 420px; flex-shrink: 0; overflow-y: auto;
    }
    .msg {
      padding: 10px 14px; margin-bottom: 8px;
      border-radius: 8px; font-size: 14px;
      line-height: 1.5; white-space: pre-wrap;
      word-break: break-word;
    }
    .msg-ai {
      background: var(--msg-ai);
      border: 1px solid var(--msg-ai-border);
    }
    .msg-human {
      background: var(--msg-human);
      border: 1px solid var(--msg-human-border);
    }
    .msg-tool {
      background: var(--msg-tool);
      border: 1px solid var(--msg-tool-border);
      font-family: monospace; font-size: 12px;
    }
    .msg-label {
      font-size: 11px; font-weight: 600; color: var(--text3);
      margin-bottom: 4px; text-transform: uppercase;
    }
    .turn-label {
      font-size: 10px; color: var(--text3); float: right;
      font-weight: 400;
    }
    .tool-expand-btn {
      display: inline-block; margin-top: 6px;
      font-size: 11px; color: var(--accent); cursor: pointer;
      border: none; background: none; padding: 0;
    }
    .tool-expand-btn:hover { text-decoration: underline; }
    #llm-scores-panel, #annotation-panel {
      background: var(--card); border-radius: 8px; padding: 16px;
      border: 1px solid var(--card-border);
      margin-bottom: 12px;
    }
    #llm-scores-panel h3, #annotation-panel h3 {
      font-size: 14px; margin-bottom: 12px;
    }
    .score-row {
      display: flex; justify-content: space-between;
      align-items: center; padding: 6px 0;
      border-bottom: 1px solid var(--card-border);
    }
    .score-dim { font-size: 14px; font-weight: 500; }
    .score-val {
      font-size: 14px; font-weight: 700;
      padding: 2px 8px; border-radius: 4px;
    }
    .score-1 { background: #fee2e2; color: #991b1b; }
    .score-2 { background: #ffedd5; color: #9a3412; }
    .score-3 { background: #fef9c3; color: #854d0e; }
    .score-4 { background: #dcfce7; color: #166534; }
    .score-5 { background: #d1fae5; color: #065f46; }
    .score-explanation {
      font-size: 13px; color: var(--text3);
      margin-top: 4px; padding-left: 8px;
      line-height: 1.5;
    }
    .score-explanation.truncated {
      max-height: 7.5em; overflow: hidden;
      position: relative;
    }
    .expl-expand-btn {
      display: inline-block; margin-top: 2px;
      font-size: 11px; color: var(--accent); cursor: pointer;
      border: none; background: none; padding: 0;
    }
    .expl-expand-btn:hover { text-decoration: underline; }
    """

    def _css4(self) -> str:
        return """
    .annotation-field { margin-bottom: 12px; }
    .annotation-field label {
      display: block; font-size: 12px; font-weight: 600;
      margin-bottom: 4px; color: var(--text2);
    }
    .annotation-field input,
    .annotation-field textarea,
    .annotation-field select {
      width: 100%; padding: 6px 10px;
      border: 1px solid var(--input-border); border-radius: 4px;
      font-size: 13px; background: var(--input-bg);
      color: var(--text);
    }
    .dim-score-group { margin-bottom: 14px; }
    .dim-score-header {
      display: flex; justify-content: space-between;
      align-items: center; margin-bottom: 2px;
    }
    .dim-score-header label {
      font-size: 14px; font-weight: 600; color: var(--text);
    }
    .dim-score-header select {
      width: 70px; flex-shrink: 0;
      padding: 4px 6px; border: 1px solid var(--input-border);
      border-radius: 4px; font-size: 14px;
      background: var(--input-bg); color: var(--text);
    }
    .rubric-guide {
      font-size: 13px; color: var(--text2); line-height: 1.4;
      padding-left: 4px; margin-top: 1px;
    }
    .rubric-guide div { margin-bottom: 0; }
    .dim-comment {
      width: 100%; padding: 4px 8px; margin-top: 4px;
      border: 1px solid var(--input-border); border-radius: 4px;
      font-size: 12px; background: var(--input-bg);
      color: var(--text); resize: vertical;
      font-family: inherit;
    }
    .dim-comment::placeholder { color: var(--text2); opacity: 0.6; }
    .scoring-instruction {
      font-size: 12px; color: var(--text2); font-style: italic;
      padding: 8px; margin-bottom: 12px;
      border-left: 3px solid var(--accent);
      background: var(--card-bg); border-radius: 0 4px 4px 0;
      line-height: 1.4;
    }
    .profile-section { margin-bottom: 10px; }
    .profile-section-title {
      font-size: 13px; font-weight: 700; color: var(--accent);
      text-transform: capitalize; margin-bottom: 4px;
      border-bottom: 1px solid var(--border); padding-bottom: 2px;
    }
    .profile-row {
      font-size: 12px; line-height: 1.5; padding: 1px 0;
      color: var(--text);
    }
    .profile-key {
      font-weight: 600; color: var(--text2);
      text-transform: capitalize;
    }
    .profile-val { color: var(--text); }
    #submit-annotation {
      width: 100%; padding: 10px; background: #22c55e;
      color: #fff; border: none; border-radius: 6px;
      cursor: pointer; font-size: 14px; font-weight: 600;
    }
    #submit-annotation:hover { background: #16a34a; }
    #annotation-status {
      margin-top: 8px; font-size: 13px; text-align: center;
    }
    .hidden { display: none !important; }
    """


    def _js(
        self, conv_data: str, dims_json: str, rubric_dims_json: str,
    ) -> str:
        return self._js_data(conv_data, dims_json, rubric_dims_json) + \
            self._js_init() + self._js_show() + \
            self._js_actions() + self._js_util()

    def _js_data(
        self, conv_data: str, dims_json: str, rubric_dims_json: str,
    ) -> str:
        rubric_guide_json = json.dumps(RUBRIC_GUIDE)
        return f"""
var conversations = {conv_data};
var ANNOTATION_DIMS = {dims_json};
var RUBRIC_DIMS = {rubric_dims_json};
var currentIndex = 0;
var annotationMode = true;
var hasUnsaved = false;
var RUBRIC_GUIDE = {rubric_guide_json};

// --- Configurable active rubrics ---
// Edit this list to limit which rubrics annotators see and must complete.
// By default all annotation dimensions are active.
// Example: set to ["clinical_safety","triage_quality","clinical_helpfulness"]
// for the clinical annotator group.
var ACTIVE_RUBRICS = ANNOTATION_DIMS.slice();
"""


    def _js_init(self) -> str:
        return """
    window.addEventListener('beforeunload', function(e) {
      if (hasUnsaved) {
    e.preventDefault();
    e.returnValue = 'You have unsaved annotations. Download before leaving?';
      }
    });

    document.addEventListener('keydown', function(e) {
      if (e.ctrlKey && e.shiftKey && e.key === 'X') {
    var btn = document.getElementById('annotation-toggle');
    var warn = document.getElementById('annotation-warn');
    var show = btn.style.display === 'none';
    btn.style.display = show ? '' : 'none';
    warn.style.display = show ? 'inline' : 'none';
      }
    });

    function init() {
      var saved = localStorage.getItem('pab_annotator_id');
      if (saved) document.getElementById('annotator-id').value = saved;
      var container = document.getElementById('annotation-scores');
      var instrDiv = document.createElement('div');
      instrDiv.className = 'scoring-instruction';
      instrDiv.textContent = 'Score the highest level where ALL requirements ' +
    'at that level and below are met. Partial credit at a higher level ' +
    'does not override a missed requirement at a lower level.';
      container.appendChild(instrDiv);
      ANNOTATION_DIMS.forEach(function(dim) {
    var group = document.createElement('div');
    group.className = 'dim-score-group';
    group.setAttribute('data-dim', dim);
    if (ACTIVE_RUBRICS.indexOf(dim) === -1) group.style.display = 'none';
    var header = document.createElement('div');
    header.className = 'dim-score-header';
    var label = document.createElement('label');
    label.textContent = dim.replace(/_/g, ' ');
    var sel = document.createElement('select');
    sel.id = 'score-' + dim;
    sel.innerHTML = '<option value="">--</option>';
    for (var s = 1; s <= 5; s++)
      sel.innerHTML += '<option value="'+s+'">'+s+'</option>';
    sel.addEventListener('change', autosaveDraft);
    header.appendChild(label);
    header.appendChild(sel);
    group.appendChild(header);
    var guide = RUBRIC_GUIDE[dim];
    if (guide) {
      var gDiv = document.createElement('div');
      gDiv.className = 'rubric-guide';
      guide.forEach(function(line) {
        var d = document.createElement('div');
        d.textContent = line;
        gDiv.appendChild(d);
      });
      group.appendChild(gDiv);
    }
    var commentTA = document.createElement('textarea');
    commentTA.id = 'comment-' + dim;
    commentTA.className = 'dim-comment';
    commentTA.rows = 2;
    commentTA.placeholder = 'Justification for this score (recommended)';
    commentTA.addEventListener('input', autosaveDraft);
    group.appendChild(commentTA);
    container.appendChild(group);
      });
      updateNavExpVisibility();
      updateAllNavChecks();
      if (conversations.length > 0) showConversation(0);
    }
    """


    def _js_show(self) -> str:
        return """
    function showConversation(idx) {
      currentIndex = idx;
      var conv = conversations[idx];
      document.querySelectorAll('.nav-item').forEach(function(el, i) {
    el.classList.toggle('active', i === idx);
      });
      updateConvTitle(conv);
      document.getElementById('scenario-content').textContent =
    conv.scenario;
      renderProfile(conv.patient_profile);
      renderMessages(conv.conversation);
      renderLLMScores(conv);
      prefillAnnotation(conv);
      document.getElementById('annotation-status').textContent = '';
    }

    function updateConvTitle(conv) {
      var el = document.getElementById('conv-title');
      if (annotationMode) {
    el.textContent = conv.case_id;
      } else {
    el.textContent = conv.case_id + ' \\u2014 ' +
      conv.experiment.experiment_id +
      ' (' + conv.experiment.assistant_label + ')';
      }
    }

    function parseXmlProfile(xml) {
      var sections = [];
      var tagStack = [];
      var lines = xml.split('\\n');
      var currentSection = null;
      var currentKey = null;
      var currentValue = [];

      for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();
    if (!line) continue;
    var openMatch = line.match(/^<([a-zA-Z_][a-zA-Z0-9_-]*)>$/);
    var closeMatch = line.match(/^<\\/([a-zA-Z_][a-zA-Z0-9_-]*)>$/);
    var selfContent = line.match(/^<([a-zA-Z_][a-zA-Z0-9_-]*)>(.+)<\\/\\1>$/);

    if (selfContent) {
      var key = selfContent[1].replace(/_/g, ' ');
      var val = selfContent[2].trim();
      if (currentSection) {
        currentSection.items.push({key: key, value: val});
      } else {
        sections.push({title: null, items: [{key: key, value: val}]});
      }
    } else if (openMatch && !closeMatch) {
      var tag = openMatch[1];
      if (tagStack.length === 0 || (tagStack.length === 1 && !currentKey)) {
        currentSection = {title: tag.replace(/_/g, ' '), items: []};
        sections.push(currentSection);
      } else {
        currentKey = tag.replace(/_/g, ' ');
        currentValue = [];
      }
      tagStack.push(tag);
    } else if (closeMatch) {
      tagStack.pop();
      if (currentKey && tagStack.length <= 1) {
        if (currentValue.length > 0) {
          currentSection.items.push({
            key: currentKey,
            value: currentValue.join(', ')
          });
        }
        currentKey = null;
        currentValue = [];
      }
      if (tagStack.length === 0) {
        currentSection = null;
      }
    } else if (line && currentKey) {
      currentValue.push(line);
    } else if (line && currentSection) {
      currentSection.items.push({key: '', value: line});
    }
      }
      return sections;
    }

    function renderProfile(xml) {
      var el = document.getElementById('patient-profile-content');
      var sections = parseXmlProfile(xml);
      if (sections.length === 0) {
    el.innerHTML = '<pre style="margin:0;white-space:pre-wrap;' +
      'word-break:break-word;font-size:12px;">' +
      escapeHtml(xml) + '</pre>';
    return;
      }
      var h = '';
      sections.forEach(function(sec) {
    if (sec.title) {
      h += '<div class="profile-section">';
      h += '<div class="profile-section-title">' +
        escapeHtml(sec.title) + '</div>';
    }
    sec.items.forEach(function(item) {
      if (item.key) {
        h += '<div class="profile-row"><span class="profile-key">' +
          escapeHtml(item.key) + ':</span> ' +
          '<span class="profile-val">' +
          escapeHtml(item.value) + '</span></div>';
      } else {
        h += '<div class="profile-row">' +
          escapeHtml(item.value) + '</div>';
      }
    });
    if (sec.title) h += '</div>';
      });
      el.innerHTML = h;
    }

    function renderMessages(msgs) {
      var h = '';
      var turnNum = 0;
      var subIdx = 0;
      for (var i = 0; i < msgs.length; i++) {
    var msg = msgs[i];
    var cls = 'msg msg-' + msg.type;
    var labelText = msg.type.toUpperCase();
    var turnLabel = '';
    if (msg.type === 'human') {
      turnNum++;
      subIdx = 0;
      turnLabel = 'Turn ' + turnNum;
    } else {
      subIdx++;
      turnLabel = turnNum + '.' + subIdx;
    }
    var content = escapeHtml(msg.content);
    if (msg.type === 'tool') {
      content = truncateContent(content, 20, 'tool-' + i);
    }
    h += '<div class="' + cls + '">' +
      '<div class="msg-label">' + labelText +
      '<span class="turn-label">' + turnLabel +
      '</span></div>' + content + '</div>';
      }
      document.getElementById('conversation-messages').innerHTML = h;
    }

    function truncateContent(text, maxLines, id) {
      var lines = text.split('\\n');
      if (lines.length <= maxLines) return text;
      var shown = lines.slice(0, maxLines).join('\\n');
      var rest = lines.slice(maxLines).join('\\n');
      return shown +
    '<div id="trunc-rest-' + id + '" style="display:none">' +
    rest + '</div>' +
    '<button class="tool-expand-btn" ' +
    'onclick="toggleTrunc(\\'' + id + '\\',this)">' +
    '... show ' + (lines.length - maxLines) +
    ' more lines</button>';
    }

    function toggleTrunc(id, btn) {
      var el = document.getElementById('trunc-rest-' + id);
      if (el.style.display === 'none') {
    el.style.display = 'inline';
    btn.textContent = 'show less';
      } else {
    el.style.display = 'none';
    var lines = el.innerHTML.split('\\n').length;
    btn.textContent = '... show ' + lines + ' more lines';
      }
    }

    function autosaveDraft() {
      var conv = conversations[currentIndex];
      var aid = document.getElementById('annotator-id').value.trim();
      var ann = conv.annotations && conv.annotations.length > 0
    ? conv.annotations[conv.annotations.length - 1] : null;
      if (!ann || ann.submitted) {
    ann = {annotator_id: aid, scores: {}, comments: {}, comment: '', submitted: false};
    conv.annotations.push(ann);
      }
      if (aid) ann.annotator_id = aid;
      ANNOTATION_DIMS.forEach(function(dim) {
    var val = document.getElementById('score-' + dim).value;
    if (val && val >= 1 && val <= 5) ann.scores[dim] = parseInt(val);
    else delete ann.scores[dim];
    var cmt = document.getElementById('comment-' + dim);
    if (cmt && cmt.value.trim()) ann.comments[dim] = cmt.value.trim();
    else delete ann.comments[dim];
      });
      hasUnsaved = true;
      updateNavCheck(currentIndex);
    }

    function prefillAnnotation(conv) {
      var ann = conv.annotations && conv.annotations.length > 0
    ? conv.annotations[conv.annotations.length - 1] : null;
      ANNOTATION_DIMS.forEach(function(dim) {
    var sel = document.getElementById('score-' + dim);
    if (ann && ann.scores && ann.scores[dim] != null) {
      sel.value = ann.scores[dim];
    } else {
      sel.value = '';
    }
    var cmt = document.getElementById('comment-' + dim);
    if (cmt) {
      cmt.value = (ann && ann.comments && ann.comments[dim]) ? ann.comments[dim] : '';
    }
      });
    }
    """


    def _js_actions(self) -> str:
        return """
    function renderLLMScores(conv) {
      var h = '';
      RUBRIC_DIMS.forEach(function(dim) {
    var rs = conv.llm_scores[dim];
    if (!rs) return;
    var scoreClass = 'score-' + Math.round(rs.score);
    h += '<div class="score-row">' +
      '<span class="score-dim">' +
      dim.replace(/_/g, ' ') + '</span>' +
      '<span class="score-val ' + scoreClass + '">' +
      rs.score + '/5</span></div>';
    if (rs.explanation) {
      var expl = escapeHtml(rs.explanation);
      h += '<div class="score-explanation truncated" ' +
        'id="expl-' + dim + '">' + expl + '</div>';
      h += '<button class="expl-expand-btn" ' +
        'id="expl-btn-' + dim + '" ' +
        'onclick="toggleExpl(\\'' + dim + '\\',this)">' +
        'show more</button>';
    }
      });
      document.getElementById('llm-scores-content').innerHTML = h;
      RUBRIC_DIMS.forEach(function(dim) {
    var el = document.getElementById('expl-' + dim);
    var btn = document.getElementById('expl-btn-' + dim);
    if (el && btn) {
      if (el.scrollHeight <= el.clientHeight + 2) {
        btn.style.display = 'none';
        el.classList.remove('truncated');
      }
    }
      });
    }

    function toggleExpl(dim, btn) {
      var el = document.getElementById('expl-' + dim);
      if (el.classList.contains('truncated')) {
    el.classList.remove('truncated');
    btn.textContent = 'show less';
      } else {
    el.classList.add('truncated');
    btn.textContent = 'show more';
      }
    }

    function promptExitAnnotation() {
      var pw = prompt('Enter reviewer password to exit annotation mode:');
      if (pw === 'pab2026') {
    toggleAnnotationMode();
      } else if (pw !== null) {
    alert('Incorrect password.');
      }
    }

    function toggleAnnotationMode() {
      annotationMode = !annotationMode;
      var btn = document.getElementById('annotation-toggle');
      if (annotationMode) {
    btn.textContent = 'Exit Annotation Mode';
    btn.classList.remove('inactive');
      } else {
    btn.textContent = 'Enter Annotation Mode';
    btn.classList.add('inactive');
      }
      var llm = document.getElementById('llm-scores-panel');
      var ann = document.getElementById('annotation-panel');
      var warn = document.getElementById('annotation-warn');
      llm.classList.toggle('hidden', annotationMode);
      ann.classList.toggle('hidden', !annotationMode);
      warn.style.display = annotationMode ? 'inline' : 'none';
      updateNavExpVisibility();
      updateConvTitle(conversations[currentIndex]);
      if (!annotationMode) renderLLMScores(conversations[currentIndex]);
    }

    function updateNavExpVisibility() {
      var navExps = document.querySelectorAll('.nav-exp');
      navExps.forEach(function(el) {
    el.style.display = annotationMode ? 'none' : 'block';
      });
    }

    function updateAllNavChecks() {
      for (var i = 0; i < conversations.length; i++) {
    updateNavCheck(i);
      }
    }

    function updateNavCheck(idx) {
      var el = document.getElementById('nav-check-' + idx);
      if (!el) return;
      var conv = conversations[idx];
      var lastAnn = conv.annotations && conv.annotations.length > 0
    ? conv.annotations[conv.annotations.length - 1] : null;
      if (lastAnn && lastAnn.submitted) {
    el.textContent = '\u2713';
    el.className = 'nav-check done';
      } else if (lastAnn && !lastAnn.submitted && Object.keys(lastAnn.scores || {}).length > 0) {
    el.textContent = '\u270E';
    el.className = 'nav-check draft';
      } else {
    el.textContent = '';
    el.className = 'nav-check';
      }
    }

    function toggleTheme() {
      document.body.classList.toggle('dark');
      document.body.classList.toggle('light');
      var btn = document.getElementById('theme-toggle');
      if (document.body.classList.contains('light')) {
    btn.innerHTML = '&#9789;';
      } else {
    btn.innerHTML = '&#9788;';
      }
    }
    """


    def _js_util(self) -> str:
        return """
    function submitAnnotation() {
      var aid = document.getElementById('annotator-id').value.trim();
      if (!aid) { setStatus('Annotator ID is required.','red'); return; }
      localStorage.setItem('pab_annotator_id', aid);
      var conv = conversations[currentIndex];
      var ann = conv.annotations && conv.annotations.length > 0
    ? conv.annotations[conv.annotations.length - 1] : null;
      if (!ann || ann.submitted) {
    ann = {annotator_id: aid, scores: {}, comments: {}, comment: '', submitted: false};
    conv.annotations.push(ann);
      }
      var valid = true;
      ANNOTATION_DIMS.forEach(function(dim) {
    var val = document.getElementById('score-' + dim).value;
    var isActive = ACTIVE_RUBRICS.indexOf(dim) !== -1;
    if (isActive && (!val || val < 1 || val > 5)) valid = false;
    if (val && val >= 1 && val <= 5) ann.scores[dim] = parseInt(val);
    var cmt = document.getElementById('comment-' + dim);
    if (cmt && cmt.value.trim()) {
      ann.comments[dim] = cmt.value.trim();
    } else {
      delete ann.comments[dim];
    }
      });
      if (!valid) {
    setStatus('All active rubric scores (1-5) are required.', 'red');
    return;
      }
      ann.annotator_id = aid;
      ann.submitted = true;
      ann.timestamp = new Date().toISOString().replace(
    /[-:T]/g, '').slice(0, 15).replace(
    /^(.{8})/, '$1_');
      hasUnsaved = true;
      updateNavCheck(currentIndex);
      setStatus('Annotation saved. Remember to download.', 'green');
    }

    function downloadAnnotations() {
      var blob = new Blob(
    [JSON.stringify(conversations, null, 2)],
    { type: 'application/json' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'annotations.json';
      a.click();
      URL.revokeObjectURL(url);
      hasUnsaved = false;
    }

    function setStatus(msg, color) {
      var el = document.getElementById('annotation-status');
      el.textContent = msg;
      el.style.color = color;
    }

    function escapeHtml(text) {
      if (!text) return '';
      var div = document.createElement('div');
      div.appendChild(document.createTextNode(text));
      return div.innerHTML;
    }

    init();
    """
