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

  // Clinique Amina wordmark + monogram, replacing OHIF's default logo in the
  // WorkList toolbar — the sanctioned whiteLabeling extension point
  // (WorkList.tsx reads this directly), no core edit needed. Plain
  // React.createElement (not JSX): this file is loaded as a raw <script>, not
  // run through a JSX transform — see the commented example this follows in
  // config/kheops.js. font-serif resolves to Playfair Display
  // (tailwind.config.js + the Google Fonts link in html-templates/index.html);
  // the gold-ringed monogram matches the same mark used at the top of the
  // findings panel (DoctorAssistantPanel.tsx's PanelBrandHeader) so the two
  // don't read as two different products.
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
              'border-accent bg-secondary text-primary shadow-brand-sm flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2',
          },
          React.createElement(
            'span',
            { className: 'font-serif text-sm font-bold' },
            'A'
          )
        ),
        React.createElement(
          'span',
          { className: 'flex flex-col leading-tight' },
          React.createElement(
            'span',
            { className: 'font-serif text-foreground text-[15px] font-semibold tracking-tight' },
            'Clinique Amina'
          ),
          React.createElement(
            'span',
            { className: 'text-muted-foreground text-[10.5px] tracking-wide uppercase' },
            'AI Imaging Review'
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
  doctorAssistantApiBaseUrl: 'http://localhost:8000',

  // TEMPORARY for UI/theme preview: defaults to the public demo dataset
  // (no Orthanc/Docker required) so the viewer has real studies to show
  // while Phase 2's Orthanc container isn't running in this environment.
  // Switch back to 'orthancProxy' once Orthanc is up for real end-to-end
  // testing — both sources stay configured below either way.
  defaultDataSourceName: 'demoDataSource',
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
