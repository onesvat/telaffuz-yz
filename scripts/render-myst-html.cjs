const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..');
const thesisDir = path.join(repoRoot, 'thesis');
const siteDir = path.join(thesisDir, '_build', 'site');
const htmlDir = path.join(thesisDir, '_build', 'html');
const templatePublicDir = path.join(thesisDir, 'templates', 'myst-offline-html', 'public');

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderInline(node) {
  if (!node) return '';
  if (node.type === 'text') return escapeHtml(node.value);
  if (node.type === 'emphasis') return `<em>${renderChildren(node)}</em>`;
  if (node.type === 'strong') return `<strong>${renderChildren(node)}</strong>`;
  if (node.type === 'inlineCode') return `<code>${escapeHtml(node.value)}</code>`;
  if (node.type === 'link') return `<a href="${escapeHtml(node.url)}">${renderChildren(node)}</a>`;
  if (node.type === 'cite') return renderChildren(node);
  if (node.type === 'citeGroup') {
    const cites = (node.children || []).map(renderInline).join('; ');
    return node.kind === 'narrative' ? cites : `(${cites})`;
  }
  return renderChildren(node);
}

function renderChildren(node, context = {}) {
  return (node.children || []).map((child) => renderNode(child, context)).join('');
}

function renderBibliography(allReferences) {
  const entries = Array.from(allReferences.values());
  if (!entries.length) return '<p>Kaynak girdisi bulunamadı.</p>';
  return `<ol class="bibliography">${entries
    .map((entry) => `<li>${entry.html || escapeHtml(entry.label)}</li>`)
    .join('')}</ol>`;
}

function renderNode(node, context = {}) {
  if (!node) return '';
  if (node.type === 'root' || node.type === 'block') return renderChildren(node, context);
  if (node.type === 'paragraph') return `<p>${renderChildren(node, context)}</p>`;
  if (node.type === 'heading') {
    const depth = Math.min(Math.max(Number(node.depth || 2), 1), 6);
    return `<h${depth}>${renderChildren(node, context)}</h${depth}>`;
  }
  if (node.type === 'bibliography') return renderBibliography(context.allReferences || new Map());
  return renderInline(node);
}

function projectPages(config) {
  const project = (config.projects || [])[0] || {};
  return [
    { slug: 'index', title: 'Giriş', output: 'index.html' },
    ...(project.pages || []).map((page) => ({
      slug: page.slug,
      title: page.title || page.slug,
      output: path.join(page.slug, 'index.html'),
    })),
  ];
}

function collectReferences(pages) {
  const refs = new Map();
  for (const page of pages) {
    const article = readJson(path.join(siteDir, 'content', `${page.slug}.json`));
    const cite = article.references && article.references.cite;
    for (const key of cite?.order || []) {
      if (cite.data?.[key] && !refs.has(key)) refs.set(key, cite.data[key]);
    }
  }
  return refs;
}

function renderNav(config, pageDir) {
  const project = (config.projects || [])[0] || {};
  const links = [
    { output: 'index.html', title: 'Giriş' },
    ...(project.pages || []).map((page) => ({
      output: path.join(page.slug, 'index.html'),
      title: page.title || page.slug,
    })),
  ];
  return `<nav class="site-nav"><h2 class="site-title">${escapeHtml(project.title || 'MyST')}</h2>${links
    .map((link) => {
      const href = path.relative(pageDir, link.output).split(path.sep).join('/') || 'index.html';
      return `<a href="${escapeHtml(href)}">${escapeHtml(link.title)}</a>`;
    })
    .join('')}</nav>`;
}

function renderPage(config, page, allReferences) {
  const article = readJson(path.join(siteDir, 'content', `${page.slug}.json`));
  const title = article.frontmatter?.title || page.title || article.slug || 'Untitled';
  const body = renderNode(article.mdast, { allReferences });
  const pageDir = path.dirname(page.output);
  const cssHref = path.relative(pageDir, 'myst-theme.css').split(path.sep).join('/') || 'myst-theme.css';
  return `<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <link rel="stylesheet" href="${escapeHtml(cssHref)}">
</head>
<body>
  <div class="site-shell">
    ${renderNav(config, pageDir)}
    <main class="site-main">
      <h1>${escapeHtml(title)}</h1>
      ${body}
    </main>
  </div>
</body>
</html>`;
}

function writeFile(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
}

function copyIfExists(src, dest) {
  if (fs.existsSync(src)) fs.cpSync(src, dest, { recursive: true });
}

const config = readJson(path.join(siteDir, 'config.json'));
const pages = projectPages(config);
const allReferences = collectReferences(pages);

fs.rmSync(htmlDir, { recursive: true, force: true });
fs.mkdirSync(htmlDir, { recursive: true });

copyIfExists(templatePublicDir, htmlDir);
copyIfExists(path.join(siteDir, 'public'), path.join(htmlDir, 'build'));

for (const filename of ['config.json', 'objects.inv', 'myst.xref.json', 'myst.search.json']) {
  copyIfExists(path.join(siteDir, filename), path.join(htmlDir, filename));
}

for (const page of pages) {
  const json = fs.readFileSync(path.join(siteDir, 'content', `${page.slug}.json`), 'utf8');
  writeFile(path.join(htmlDir, page.output), renderPage(config, page, allReferences));
  writeFile(path.join(htmlDir, `${page.slug}.json`), json);
}

writeFile(path.join(htmlDir, 'robots.txt'), '');
writeFile(path.join(htmlDir, 'sitemap.xml'), '');
writeFile(path.join(htmlDir, 'sitemap_style.xsl'), '');
writeFile(path.join(htmlDir, 'favicon.ico'), '');

console.log(`Wrote ${pages.length} HTML pages to ${path.relative(repoRoot, htmlDir)}`);
