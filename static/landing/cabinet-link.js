/* Единая точка правки адреса кабинета для страниц лендинга на GitHub Pages.
   Если приложение переедет на другой хостинг — поменяйте только эту строку. */
window.VASCULARAI_CABINET_URL = "https://med-it.streamlit.app/";

/* Открывает кабинет, передавая нужный раздел как ?page=<slug>. */
window.openCabinet = function openCabinet(slug) {
  var base = window.VASCULARAI_CABINET_URL || "/";
  var url;
  try {
    url = new URL(base, window.location.href);
  } catch (error) {
    window.location.replace(base);
    return;
  }
  if (slug) {
    url.searchParams.set("page", slug);
  }
  window.location.replace(url.toString());
};
