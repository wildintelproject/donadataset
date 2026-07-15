/* Click-to-zoom for Mermaid diagrams (flowcharts, sequence diagrams...).
 *
 * Material renders each ```mermaid fence by replacing it with a
 * <div class="mermaid"> that hosts a CLOSED shadow root containing the
 * actual <svg> — so the SVG is never reachable via querySelector from
 * outside (container.querySelector("svg") always returns null, closed
 * mode blocks .shadowRoot too). The only way to "zoom" it is to move the
 * real host element into the overlay (its shadow root travels with it),
 * not clone it, and scale the whole host via CSS transform.
 *
 * Rendering is also async, so this binds via delegation on document
 * instead of on the diagrams directly, which would miss ones rendered
 * after this script first runs. */
(function () {
  "use strict";

  var overlay = null;
  var toolbar = null;
  var active = null; // the .mermaid element currently moved into the overlay
  var placeholder = null; // marks where `active` came from, to restore it
  var scale = 1;

  function closeOverlay() {
    if (!active) return;
    active.style.transform = "";
    active.style.transformOrigin = "";
    active.style.cursor = "";
    placeholder.replaceWith(active);
    overlay.remove();
    toolbar.remove();
    overlay = null;
    toolbar = null;
    active = null;
    placeholder = null;
    scale = 1;
    document.body.style.overflow = "";
  }

  function setScale(next) {
    scale = Math.min(Math.max(next, 0.25), 8);
    active.style.transform = "scale(" + scale + ")";
  }

  // Scales the diagram up (SVG, so this stays crisp) to fill as much of the
  // overlay as it can without overflowing — called once on open, and again
  // by the "Fit" button. Derives the natural (unscaled) size by dividing the
  // current rect by the current `scale` instead of clearing the transform
  // and re-measuring: that used to race the CSS transition (reading the
  // rect mid-transition, before it settled at the cleared value) and gave
  // wildly wrong numbers.
  function fitToScreen() {
    var rect = active.getBoundingClientRect();
    var naturalWidth = rect.width / scale;
    var naturalHeight = rect.height / scale;
    var reserved = 96; // toolbar + breathing room, matches the overlay's own padding
    var availableWidth = window.innerWidth - reserved;
    var availableHeight = window.innerHeight - reserved;
    var fit = Math.min(availableWidth / naturalWidth, availableHeight / naturalHeight);
    setScale(fit);
  }

  function makeButton(label, title, onClick) {
    var button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.title = title;
    button.addEventListener("click", onClick);
    return button;
  }

  function openOverlay(container) {
    active = container;
    placeholder = document.createComment("mermaid-zoom-placeholder");
    container.before(placeholder);

    overlay = document.createElement("div");
    overlay.className = "mermaid-zoom-overlay";
    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) closeOverlay();
    });

    container.style.transformOrigin = "top center";
    container.style.cursor = "default";
    overlay.appendChild(container);

    toolbar = document.createElement("div");
    toolbar.className = "mermaid-zoom-toolbar";
    toolbar.appendChild(makeButton("−", "Zoom out", function () { setScale(scale - 0.25); }));
    toolbar.appendChild(makeButton("+", "Zoom in", function () { setScale(scale + 0.25); }));
    toolbar.appendChild(makeButton("⤢", "Fit to screen", fitToScreen));
    toolbar.appendChild(makeButton("✕", "Close", closeOverlay));

    document.body.appendChild(overlay);
    document.body.appendChild(toolbar);
    document.body.style.overflow = "hidden";

    fitToScreen();
  }

  document.addEventListener("click", function (event) {
    if (overlay) return; // already zoomed in; overlay's own handlers deal with closing
    if (event.target.closest(".mermaid-zoom-toolbar")) return;
    var container = event.target.closest(".mermaid");
    if (!container) return;
    openOverlay(container);
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeOverlay();
  });
})();
