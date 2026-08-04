#!/usr/bin/env node
// Workaround for a pnpm@12.0.0-beta.3 `nodeLinker: hoisted` bug where many
// direct workspace dependencies, and some transitive version-conflict pairs
// (e.g. rspack@2.0.3's private binding, ajv-keywords@3 vs the root's ajv@8),
// never get materialized in the consuming package's own node_modules despite
// being present in the lockfile and the root hoist. Run after `pnpm install`
// whenever you see "Cannot find module", "Module not found", or an
// ajv-keywords "reading 'date' of undefined" crash. Idempotent.
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const ROOT_NM = path.join(ROOT, 'node_modules');

function symlinkIfMissing(localPath, targetPath) {
  if (fs.existsSync(localPath)) return false;
  if (!fs.existsSync(targetPath)) return false;
  fs.mkdirSync(path.dirname(localPath), { recursive: true });
  fs.symlinkSync(path.relative(path.dirname(localPath), targetPath), localPath);
  return true;
}

function fixDirectDependencyGaps() {
  const workspaceGlobs = ['platform', 'extensions', 'modes'];
  const pkgDirs = [];
  for (const group of workspaceGlobs) {
    const groupDir = path.join(ROOT, group);
    if (!fs.existsSync(groupDir)) continue;
    for (const entry of fs.readdirSync(groupDir, { withFileTypes: true })) {
      if (entry.isDirectory()) pkgDirs.push(path.join(group, entry.name));
    }
  }

  let total = 0;
  for (const dir of pkgDirs) {
    const pkgPath = path.join(ROOT, dir, 'package.json');
    if (!fs.existsSync(pkgPath)) continue;
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    const deps = Object.assign({}, pkg.dependencies, pkg.devDependencies, pkg.peerDependencies);
    const localNM = path.join(ROOT, dir, 'node_modules');
    let fixedHere = [];
    for (const [name, spec] of Object.entries(deps)) {
      if (String(spec).startsWith('workspace:')) continue;
      const fixed = symlinkIfMissing(path.join(localNM, name), path.join(ROOT_NM, name));
      if (fixed) fixedHere.push(name);
    }
    if (fixedHere.length) {
      console.log(dir + ': linked ' + fixedHere.length + ' missing dep(s) -> ' + fixedHere.join(', '));
      total += fixedHere.length;
    }
  }
  return total;
}

function fixRspackBindingNesting() {
  const nestedCore = path.join(ROOT_NM, '@rspack/cli/node_modules/@rspack/core');
  if (!fs.existsSync(nestedCore)) return;
  const nestedBindingDir = path.join(nestedCore, 'node_modules/@rspack');
  if (fs.existsSync(path.join(nestedBindingDir, 'binding'))) return;

  console.warn(
    'fix-pnpm-hoist-gaps: @rspack/core@2.0.3 is missing its private @rspack/binding@2.0.3.\n' +
      '  This cannot be auto-fixed from local files (the matching native binding was never\n' +
      '  downloaded by pnpm). Fetch it manually once:\n' +
      '    npm pack @rspack/binding@2.0.3 @rspack/binding-linux-x64-gnu@2.0.3 -o /tmp/rspack-fix\n' +
      '  then extract both tarballs into:\n' +
      '    ' + nestedBindingDir + '/binding/ and ' + nestedBindingDir + '/binding-linux-x64-gnu/\n' +
      '  (adjust the platform suffix for non-linux-x64 machines).'
  );
}

function fixAjvKeywordsPairing() {
  const goodAjv6 = path.join(ROOT_NM, 'ajv-keywords/node_modules/ajv');
  if (!fs.existsSync(goodAjv6)) return 0;

  let fixed = 0;
  const loaderNames = fs.existsSync(ROOT_NM)
    ? fs.readdirSync(ROOT_NM).filter(name => !name.startsWith('.') && !name.startsWith('@'))
    : [];
  for (const name of loaderNames) {
    const nestedSchemaUtils = path.join(ROOT_NM, name, 'node_modules/schema-utils/package.json');
    if (!fs.existsSync(nestedSchemaUtils)) continue;
    const version = JSON.parse(fs.readFileSync(nestedSchemaUtils, 'utf8')).version;
    if (!/^[23]\./.test(version)) continue;
    if (symlinkIfMissing(path.join(ROOT_NM, name, 'node_modules/ajv'), goodAjv6)) {
      console.log('ajv pairing: linked ' + name + '/node_modules/ajv -> ajv@6.15.0 (schema-utils@' + version + ')');
      fixed++;
    }
  }
  return fixed;
}

const depsFixed = fixDirectDependencyGaps();
fixRspackBindingNesting();
const ajvFixed = fixAjvKeywordsPairing();
console.log('\nDone. ' + depsFixed + ' direct-dependency link(s), ' + ajvFixed + ' ajv pairing(s) fixed.');
