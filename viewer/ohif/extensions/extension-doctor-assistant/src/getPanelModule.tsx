import React from 'react';
import DoctorAssistantPanel from './DoctorAssistantPanel';

export default function getPanelModule() {
  return [
    {
      name: 'findingsPanel',
      iconName: 'tab-4d',
      iconLabel: 'AI review',
      label: 'AI Review',
      component: () => <DoctorAssistantPanel />,
    },
  ];
}
