/**
 * Registered as a `name: 'default'` customization module, same convention as
 * extensions/default and extensions/cornerstone's own getCustomizationModule —
 * CustomizationService.init() merges every registered extension's `default`-named
 * module into Default scope, in extension-registration order (this extension is
 * listed last in pluginConfig.json), so the $apply below composes against
 * extensions/default's already-registered plain-array 'workList.columns' value
 * rather than racing it.
 *
 * The toolbar/overlay entries below were originally nested inside the mode's own
 * `doctorAssistantModeCustomizations` block ($apply commands keyed by *other*
 * customization ids, e.g. 'cornerstone.toolbarButtons'). That crashed at appInit:
 * appInit.js eagerly registers every mode's `customizations` export at Default
 * scope, keyed by its own top-level name — since nothing existed yet under
 * 'doctorAssistantModeCustomizations', immutability-helper tried to walk into
 * `undefined['cornerstone.toolbarButtons']` and threw. $apply/$set only work
 * against an *existing* top-level customization id, which is exactly what
 * registering them here (Default scope, after cornerstone's own registration)
 * provides.
 */
export default function getCustomizationModule() {
  return [
    {
      name: 'default',
      value: {
        // Keep conventional radiology language and append the MedSAMBox
        // interactive-segmentation button (registered by
        // tools/registerMedSAMBoxTool.ts, activated via cornerstone's own
        // setToolActiveToolbar/evaluate.cornerstoneTool -- no new command needed).
        'cornerstone.toolbarButtons': {
          $apply: (buttons: any[]) => [
            ...buttons.map(btn =>
              btn.id === 'WindowLevel'
                ? { ...btn, props: { ...btn.props, label: 'Window / Level' } }
                : btn
            ),
            {
              id: 'MedSAMBox',
              uiType: 'ohif.toolButton',
              props: {
                type: 'tool',
                icon: 'tool-rectangle',
                label: 'Refine with box',
                // Mirrors extensions/cornerstone/src/customizations/toolbarButtonsCustomization.ts's
                // own `setToolActiveToolbar` const shape exactly (same toolGroupIds this
                // extension's registerMedSAMBoxTool.ts adds MedSAMBox to).
                commands: {
                  commandName: 'setToolActiveToolbar',
                  commandOptions: { toolGroupIds: ['default', 'mpr', 'SRToolGroup', 'volume3d'] },
                },
                evaluate: 'evaluate.cornerstoneTool',
              },
            },
          ],
        },
      },
    },
  ];
}
