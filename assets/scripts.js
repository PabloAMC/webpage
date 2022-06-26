$(function () {

  // Footnotes tooltips
  for (let link of document.querySelectorAll("a[rel='footnote']")) {
    let id = link.getAttribute('href').substr(1);
    let footnote = document.getElementById(id);
    let content = footnote.children[0].cloneNode(true);
    content.classList.add('footnote-tooltip');
    content.querySelectorAll('.reversefootnote').forEach(x => x.remove());

    link.setAttribute('title', content.innerHTML);
  };

  $('[rel="footnote"]').tooltip({
    html: true,
    placement: 'bottom',
    animation: false,
    //delay: { "show": 0, "hide": 100000000 },
  });

  // Rest of tooltips
  $('[data-toggle="tooltip"]').tooltip({
    animation: false,
  })
})

