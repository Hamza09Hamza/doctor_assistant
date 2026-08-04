import React from 'react';
import DoctorAssistantPanel from './DoctorAssistantPanel';

export default function getPanelModule() {
  return [
    {
      name: 'findingsPanel',
      iconName: 'tab-patient-info',
      iconLabel: 'Findings',
      label: 'AI Findings',
      component: () => <DoctorAssistantPanel />,
    },
  ];
}
