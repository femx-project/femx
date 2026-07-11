/* Keep the full-sidebar selection aligned with the femx main page. */
(function () {
  const page = window.location.pathname.split("/").pop();
  if (page !== "" && page !== "index.html") {
    return;
  }

  // A disabled Doxygen nav sync leaves the last anchor in local storage and
  // otherwise selects it even when a new visit displays the main page.
  try {
    window.localStorage.removeItem("navpath");
  } catch (_) {
    // Access to local storage can be disabled without affecting navigation.
  }

  function selectMainPage() {
    const links = document.querySelectorAll("#nav-tree-contents .item a");
    const link = Array.from(links).find(
      (candidate) => candidate.textContent.trim() === "femx API"
    );
    const item = link && link.closest(".item");

    if (!item || item.classList.contains("selected")) {
      return;
    }

    document
      .querySelectorAll("#nav-tree-contents .item.selected")
      .forEach((selected) => {
        selected.classList.remove("selected");
        selected.removeAttribute("id");
      });
    item.classList.add("selected");
    item.id = "selected";
  }

  document.addEventListener("DOMContentLoaded", function () {
    const navTree = document.getElementById("nav-tree-contents");
    if (!navTree) {
      return;
    }

    selectMainPage();
    new MutationObserver(selectMainPage).observe(navTree, {
      childList: true,
      subtree: true,
    });
  });
})();

/* Follow the visible documentation section in the full sidebar. */
(function () {
  function sectionIdsForPage(nodes, page, result) {
    nodes.forEach((node) => {
      const link = node[1];
      const children = node[2];
      if (typeof link === "string" && link.startsWith(page + "#")) {
        result.push(link.substring(link.indexOf("#") + 1));
      }
      if (Array.isArray(children)) {
        sectionIdsForPage(children, page, result);
      }
    });
    return result;
  }

  document.addEventListener("DOMContentLoaded", function () {
    const page = window.location.pathname.split("/").pop() || "index.html";
    const content = document.getElementById("doc-content");
    const navTree = document.getElementById("nav-tree-contents");
    if (!content || !navTree || !window.NAVTREE) {
      return;
    }

    const sectionIds = sectionIdsForPage(window.NAVTREE, page, []);
    const sections = sectionIds
      .map((id) => {
        const anchor = document.getElementById(id);
        return anchor && { id: id, heading: anchor.parentElement };
      })
      .filter(Boolean);
    if (!sections.length) {
      return;
    }

    function navLink(id) {
      const linkClass = id ? page + ":" + id : page;
      return Array.from(navTree.querySelectorAll(".item a")).find(
        (link) => link.classList.contains(linkClass)
      );
    }

    function expandPageItem() {
      const pageLink = navLink();
      const item = pageLink && pageLink.closest(".item");
      const toggle = item && item.firstElementChild;
      if (toggle && toggle.tagName === "A") {
        toggle.click();
      }
    }

    function selectLink(link) {
      const item = link && link.closest(".item");
      if (!item || item.classList.contains("selected")) {
        return;
      }

      navTree.querySelectorAll(".item.selected").forEach((selected) => {
        selected.classList.remove("selected");
        selected.removeAttribute("id");
      });
      item.classList.add("selected");
      item.id = "selected";
      item.scrollIntoView({ block: "nearest" });
    }

    function updateSelection() {
      let activeId = null;
      const maxScroll = content.scrollHeight - content.clientHeight;

      if (maxScroll > 1) {
        const contentTop = content.getBoundingClientRect().top;
        const activationPoints = sections.map((section) =>
          Math.min(
            maxScroll,
            Math.max(
              0,
              section.heading.getBoundingClientRect().top -
                contentTop +
                content.scrollTop -
                32
            )
          )
        );

        // Short final sections cannot physically reach the top offset. Give
        // each of them a small scroll interval before the bottom of the page.
        const minimumInterval = Math.min(
          48,
          maxScroll / (activationPoints.length + 1)
        );
        for (let index = activationPoints.length - 2; index >= 0; --index) {
          activationPoints[index] = Math.min(
            activationPoints[index],
            Math.max(0, activationPoints[index + 1] - minimumInterval)
          );
        }

        activationPoints.forEach((point, index) => {
          if (content.scrollTop + 1 >= point) {
            activeId = sections[index].id;
          }
        });
      }

      let link = navLink(activeId);
      if (activeId && !link) {
        expandPageItem();
        link = navLink(activeId);
      }
      selectLink(link);
    }

    let updatePending = false;
    function scheduleUpdate() {
      if (updatePending) {
        return;
      }
      updatePending = true;
      window.requestAnimationFrame(function () {
        updatePending = false;
        updateSelection();
      });
    }

    content.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    new MutationObserver(scheduleUpdate).observe(navTree, {
      childList: true,
      subtree: true,
    });
    scheduleUpdate();
  });
})();
