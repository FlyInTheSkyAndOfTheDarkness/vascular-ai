/* med-it.asia — лендинг; cabinet.med-it.asia — все кабинеты.
   Локальный лендинг подключается к локальному Streamlit. */
var localCabinet = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
window.VASCULARAI_CABINET_URL = localCabinet
  ? "http://127.0.0.1:8501/"
  : "https://cabinet.med-it.asia/";
if (window.location.hostname === "med-it.localhost") {
  window.VASCULARAI_CABINET_URL = "http://cabinet.med-it.localhost:18080/";
}

window.cabinetUrl = function cabinetUrl(slug) {
  var url = new URL(window.VASCULARAI_CABINET_URL);
  if (slug) url.searchParams.set("page", slug);
  return url.toString();
};

/* Открывает кабинет, передавая нужный раздел как ?page=<slug>. */
window.openCabinet = function openCabinet(slug) {
  window.location.replace(window.cabinetUrl(slug));
};

document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("a[data-cabinet-page]").forEach(function (link) {
    link.href = window.cabinetUrl(link.dataset.cabinetPage);
  });
});

/* Кнопка входа в готовой React-сборке должна открывать рабочий кабинет. */
document.addEventListener("click", function (event) {
  var login = event.target.closest && event.target.closest("button.text-button.login");
  if (!login) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  window.location.assign(window.cabinetUrl("dashboard"));
}, true);
