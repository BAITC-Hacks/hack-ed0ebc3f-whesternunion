from types import SimpleNamespace

from wind_agent.agent import explain


def test_ai_is_optional(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert explain({})['status'] == 'disabled'


def test_bounded_tool_call_cycle():
    invocations = []
    def create(**kwargs):
        invocations.append(kwargs)
        if len(invocations) == 1:
            return SimpleNamespace(output=[SimpleNamespace(type='function_call', name=name,
                                   call_id=name, arguments='{}') for name in
                                   ['inspect_forecast', 'inspect_data_quality']], output_text='')
        assert len([x for x in kwargs['input'] if isinstance(x, dict)
                    and x.get('type') == 'function_call_output']) == 2
        return SimpleNamespace(output=[], output_text='Проверены данные и прогноз.')
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = explain({k: {} for k in ['summary', 'weather', 'warnings', 'mode', 'quality', 'validation']}, client)
    assert result['status'] == 'complete'
    assert len(invocations) == 2
    assert all(call['store'] is False for call in invocations)
