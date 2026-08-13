/** @type {AppTypes.Config} */

// doctor_assistant local development config. The dataSource below reuses the
// already-working proxy pattern from config/docker-nginx-orthanc.js's
// `orthancProxy` entry (relative /pacs/dicom-web paths, rewritten by the dev
// server's own proxy — see the `dev:doctor-assistant` script in package.json)
// rather than pointing directly at Orthanc's origin, so the browser never has
// to cross-origin-fetch DICOMweb at all.
window.config = {
  name: 'config/doctor_assistant.js',
  routerBasename: null,
  extensions: [],
  modes: [],
  customizationService: {},
  showStudyList: true,
  maxNumberOfWebWorkers: 3,
  showWarningMessageForCrossOrigin: true,
  showCPUFallbackMessage: true,
  showLoadingIndicator: true,
  strictZSpacingForVolumeViewport: true,
  groupEnabledModesFirst: true,
  maxNumRequests: {
    interaction: 100,
    thumbnail: 5,
    prefetch: 25,
  },
  showErrorDetails: 'always',

  // Shown once (then remembered for 30 days) on first load. dialogConfiguration is
  // consumed by platform/ui-next/src/components/InvestigationalUseDialog — its copy
  // ("OHIF Viewer is ...") is partly hardcoded JSX, partly the 'InvestigationalUseDialog'
  // i18n namespace overridden in extensions/extension-doctor-assistant's preRegistration;
  // this is a second, entry-point home for the same non-clinical/experimental trust
  // messaging the findings panel leads with.
  investigationalUseDialog: {
    option: 'configure',
    days: 30,
  },

  // Clinique Amina scan-aperture mark, replacing OHIF's default logo in the
  // WorkList toolbar — the sanctioned whiteLabeling extension point
  // (WorkList.tsx reads this directly), no core edit needed. Plain
  // React.createElement (not JSX): this file is loaded as a raw <script>, not
  // run through a JSX transform — see the commented example this follows in
  // config/kheops.js. The same functional scan/pulse mark leads the clinical
  // review rail, so WorkList and viewer read as one product.
  whiteLabeling: {
    createLogoComponentFn: function (React) {
      return React.createElement(
        'a',
        {
          target: '_self',
          rel: 'noopener noreferrer',
          href: '/',
          className: 'flex items-center gap-2.5',
        },
        React.createElement(
          'span',
          {
            className:
              'border-primary/60 bg-primary/10 text-primary flex h-9 w-9 shrink-0 items-center justify-center rounded-md border',
          },
          React.createElement(
            'svg',
            {
              viewBox: '0 0 32 32',
              role: 'presentation',
              className: 'h-6 w-6 fill-none stroke-current',
              fill: 'none',
              stroke: 'currentColor',
              strokeWidth: 1.5,
              strokeLinecap: 'round',
              strokeLinejoin: 'round',
            },
            React.createElement('path', { d: 'M7 12V7h5M20 7h5v5M25 20v5h-5M12 25H7v-5' }),
            React.createElement('path', { d: 'M8 16h5l2-4 3 8 2-4h4' })
          )
        ),
        React.createElement(
          'span',
          { className: 'flex flex-col leading-tight' },
          React.createElement(
            'span',
            { className: 'text-foreground text-[14px] font-semibold tracking-tight' },
            'Clinique Amina'
          ),
          React.createElement(
            'span',
            { className: 'text-muted-foreground text-xs tracking-wide' },
            'Clinical imaging workspace'
          )
        )
      );
    },
  },

  // The doctor_assistant findings panel (extensions/extension-doctor-assistant)
  // reads this via window.config — see its apiClient.ts. Not a stock OHIF key.
  // The API itself (api/main.py) still needs its own CORS allowance for this
  // origin (OHIF_ORIGIN, default http://localhost:3000) since these are plain
  // fetch() calls to a different backend, not a DICOMweb dataSource the dev
  // server's proxy covers.
  // Relative + same-origin, proxied by the dev server to DOCTOR_ASSISTANT_API_TARGET
  // (see .webpack/webpack.pwa.js and the dev:doctor-assistant script) -- this is what
  // lets the findings panel work in a remote/sandboxed dev environment where only
  // OHIF_PORT is reachable from the browser, not the API's own port directly.
  doctorAssistantApiBaseUrl: '/doctor-assistant-api',

  // The real product path is now Orthanc. The public OHIF source remains below
  // as a manual preview fallback, but must never be confused with locally
  // generated TotalSegmentator studies.
  defaultDataSourceName: 'orthancProxy',
  dataSources: [
    {
      // Points at the Orthanc container from deployments/docker-compose.yml
      // (Phase 2) via the dev server's proxy — see dev:doctor-assistant.
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'orthancProxy',
      configuration: {
        friendlyName: 'Local Orthanc (doctor_assistant)',
        name: 'Orthanc',
        wadoUriRoot: '/wado',
        qidoRoot: '/pacs/dicom-web',
        wadoRoot: '/pacs/dicom-web',
        qidoSupportsIncludeField: false,
        imageRendering: 'wadors',
        thumbnailRendering: 'wadors',
        dicomUploadEnabled: true,
        omitQuotationForMultipartRequest: true,
      },
    },
    {
      // OHIF's own public read-only demo server (same one used by
      // config/default.js) — real DICOM studies, no local infra needed.
      // Preview-only: this is not our data and has nothing to do with the
      // doctor_assistant findings panel's AI analysis.
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'demoDataSource',
      configuration: {
        friendlyName: 'OHIF Public Demo (preview only)',
        name: 'aws',
        wadoUriRoot: 'https://d14fa38qiwhyfd.cloudfront.net/dicomweb',
        qidoRoot: 'https://d14fa38qiwhyfd.cloudfront.net/dicomweb',
        wadoRoot: 'https://d14fa38qiwhyfd.cloudfront.net/dicomweb',
        qidoSupportsIncludeField: false,
        imageRendering: 'wadors',
        thumbnailRendering: 'thumbnail',
        thumbnailRequestStrategy: 'fetch',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: false,
        supportsWildcard: true,
        staticWado: true,
        singlepart: 'bulkdata,video',
        bulkDataURI: {
          enabled: true,
          relativeResolution: 'studies',
          transform: url => url.replace('/pixeldata.mp4', '/rendered'),
        },
        omitQuotationForMultipartRequest: true,
      },
    },
  ],
  httpErrorHandler: error => {
    console.warn(error.status);
  },
};
