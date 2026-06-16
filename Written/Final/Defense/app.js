/* =====================================================================
   Defense deck — navigation, scaling, notes, overview.
   Hand-rolled, no dependencies.
   ===================================================================== */
(function () {
  "use strict";

  var stage = document.getElementById("stage");
  var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
  var counter = document.getElementById("counter");
  var progress = document.getElementById("progress");
  var notesBox = document.getElementById("notes");
  var BASE_W = 1280, BASE_H = 720;
  var cur = 0;
  var total = slides.length;

  // Wrap each slide in a .cell (keeps authoring simple; enables overview grid).
  var cells = slides.map(function (s, i) {
    var cell = document.createElement("div");
    cell.className = "cell";
    s.parentNode.insertBefore(cell, s);
    cell.appendChild(s);
    var idx = document.createElement("div");
    idx.className = "idx";
    idx.textContent = i + 1;
    cell.appendChild(idx);
    cell.addEventListener("click", function () {
      if (document.body.classList.contains("overview")) {
        toggleOverview(false);
        go(i);
      }
    });
    return cell;
  });

  function fitScale() {
    if (document.body.classList.contains("overview")) return;
    var s = Math.min(window.innerWidth / BASE_W, window.innerHeight / BASE_H);
    stage.style.transform = "translate(-50%,-50%) scale(" + s + ")";
  }

  function updateNotes() {
    var n = slides[cur].querySelector(".notes");
    notesBox.innerHTML = n ? n.innerHTML : "<h5>No notes</h5>";
  }

  function render() {
    slides.forEach(function (s, i) { s.classList.toggle("active", i === cur); });
    counter.textContent = (cur + 1) + " / " + total;
    progress.style.width = ((cur) / (total - 1) * 100) + "%";
    updateNotes();
  }

  function go(i) {
    cur = Math.max(0, Math.min(total - 1, i));
    render();
  }
  function next() { go(cur + 1); }
  function prev() { go(cur - 1); }

  function toggleOverview(force) {
    var on = (typeof force === "boolean") ? force : !document.body.classList.contains("overview");
    document.body.classList.toggle("overview", on);
    if (!on) { fitScale(); render(); }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      (document.documentElement.requestFullscreen || function () {}).call(document.documentElement);
    } else {
      (document.exitFullscreen || function () {}).call(document);
    }
  }

  // ---- keyboard ----
  document.addEventListener("keydown", function (e) {
    switch (e.key) {
      case "ArrowRight": case "PageDown": case " ": case "Spacebar":
        if (!document.body.classList.contains("overview")) { e.preventDefault(); next(); } break;
      case "ArrowLeft": case "PageUp":
        if (!document.body.classList.contains("overview")) { e.preventDefault(); prev(); } break;
      case "Home": go(0); break;
      case "End": go(total - 1); break;
      case "Escape": toggleOverview(); break;
      case "n": case "N": document.body.classList.toggle("notes"); break;
      case "f": case "F": toggleFullscreen(); break;
      default:
        if (e.key >= "0" && e.key <= "9") { /* number buffer */ buffer(e.key); }
    }
  });

  // ---- jump by typing a number then Enter ----
  var numBuf = "";
  var numTimer = null;
  function buffer(d) {
    numBuf += d;
    clearTimeout(numTimer);
    numTimer = setTimeout(function () {
      var i = parseInt(numBuf, 10);
      if (!isNaN(i)) go(i - 1);
      numBuf = "";
    }, 600);
  }

  // ---- click-halves navigation (ignore clicks on links) ----
  stage.addEventListener("click", function (e) {
    if (document.body.classList.contains("overview")) return;
    if (e.target.closest("a")) return;
    var x = e.clientX / window.innerWidth;
    if (x > 0.4) next(); else prev();
  });

  window.addEventListener("resize", fitScale);

  // ---- deep-link via #N (e.g. index.html#17) ----
  function fromHash() {
    var m = /^#(\d+)$/.exec(location.hash);
    if (m) cur = Math.max(0, Math.min(total - 1, parseInt(m[1], 10) - 1));
  }
  window.addEventListener("hashchange", function () { fromHash(); render(); });
  fromHash();

  fitScale();
  render();
})();
