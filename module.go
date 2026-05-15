package llm

import (
	"go.k6.io/k6/v2/js/common"
	"go.k6.io/k6/v2/js/modules"
)

type rootModule struct{}

func (*rootModule) NewModuleInstance(vu modules.VU) modules.Instance {
	m, err := registerMetrics(vu)
	if err != nil {
		common.Throw(vu.Runtime(), err)
	}
	return &module{vu: vu, metrics: m}
}

type module struct {
	vu      modules.VU
	metrics llmMetrics
}

func (m *module) Exports() modules.Exports {
	return modules.Exports{
		Named: map[string]any{
			"Client":  m.newClient,
			"Dataset": m.newDataset,
		},
	}
}

var _ modules.Module = (*rootModule)(nil)
