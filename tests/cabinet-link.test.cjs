const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const landing = path.join(__dirname, '../static/landing');
const source = fs.readFileSync(path.join(landing, 'cabinet-link.js'), 'utf8');

function load(hostname) {
  const events = {};
  const links = [{ dataset: { cabinetPage: 'patients' }, href: '' }];
  const location = {
    hostname,
    replace(url) { this.redirected = url; },
    assign(url) { this.assigned = url; },
  };
  const window = { location };
  vm.runInNewContext(source, {
    window, URL,
    document: {
      addEventListener(name, handler) { events[name] = handler; },
      querySelectorAll() { return links; },
    },
  });
  return { window, location, events, links };
}

test('all public cabinet routes use cabinet.med-it.asia', () => {
  for (const host of ['med-it.asia', 'www.med-it.asia']) {
    const { window, location } = load(host);
    for (const page of ['dashboard', 'intake', 'patients', 'reports', 'admin', 'settings']) {
      window.openCabinet(page);
      assert.equal(location.redirected, `https://cabinet.med-it.asia/?page=${page}`);
    }
  }
});

test('local landing opens local Streamlit for each loopback hostname', () => {
  for (const host of ['localhost', '127.0.0.1', '[::1]']) {
    const { window, location, events, links } = load(host);
    window.openCabinet('dashboard');
    assert.equal(location.redirected, 'http://127.0.0.1:8501/?page=dashboard');
    events.DOMContentLoaded();
    assert.equal(links[0].href, 'http://127.0.0.1:8501/?page=patients');
  }
});

test('header login opens the real cabinet instead of the prototype dialog', () => {
  const { location, events } = load('med-it.asia');
  let prevented = false;
  let stopped = false;
  events.click({
    target: { closest(selector) { assert.equal(selector, 'button.text-button.login'); return {}; } },
    preventDefault() { prevented = true; },
    stopImmediatePropagation() { stopped = true; },
  });
  assert.equal(location.assigned, 'https://cabinet.med-it.asia/?page=dashboard');
  assert.ok(prevented && stopped);
});

test('ordinary landing clicks do not leave the landing', () => {
  const { location, events } = load('med-it.asia');
  events.click({ target: { closest() { return null; } } });
  assert.equal(location.assigned, undefined);
});

test('local Docker domains route through the local reverse proxy', () => {
  const { window, location } = load('med-it.localhost');
  window.openCabinet('dashboard');
  assert.equal(location.redirected, 'http://cabinet.med-it.localhost:18080/?page=dashboard');
});

test('landing copy describes the actual model without unsupported headline metrics', () => {
  const html = fs.readFileSync(path.join(landing, 'index.html'), 'utf8');
  const bundle = html.match(/src="(assets\/index-[^"]+\.js)"/)[1];
  const script = fs.readFileSync(path.join(landing, bundle), 'utf8');
  assert.ok(script.includes('материнского риска'));
  assert.ok(script.includes('66,43%'));
  assert.ok(!script.includes('заявленный AUC'));
  assert.ok(!script.includes('children:"92%"'));
  assert.ok(!html.includes('ранний прогноз преэклампсии'));
});

test('HTML fallback links point to the cabinet and retain the requested page', () => {
  for (const [route, page] of Object.entries({ login: 'dashboard', cabinet: 'dashboard',
    intake: 'intake', patients: 'patients', reports: 'reports', admin: 'admin' })) {
    const html = fs.readFileSync(path.join(landing, route, 'index.html'), 'utf8');
    assert.ok(html.includes(`href="https://cabinet.med-it.asia/?page=${page}"`));
    assert.ok(html.includes(`window.openCabinet("${page}")`));
    assert.ok(!html.includes('med-it.streamlit.app'));
  }
});

test('main domain publishes the landing with a cabinet entry link', () => {
  assert.equal(fs.readFileSync(path.join(landing, 'CNAME'), 'utf8').trim(), 'med-it.asia');
  const html = fs.readFileSync(path.join(landing, 'index.html'), 'utf8');
  assert.ok(html.includes('<div id="root"></div>'));
  assert.ok(html.includes('href="https://cabinet.med-it.asia/?page=dashboard"'));
  assert.ok(html.includes('<script src="cabinet-link.js" defer></script>'));
});
