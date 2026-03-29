package tqdb

import "testing"

func TestEq(t *testing.T) {
	data := map[string]any{"name": "alice", "age": float64(30)}

	if !Eq("name", "alice").Match(data) {
		t.Error("Eq should match equal string")
	}
	if Eq("name", "bob").Match(data) {
		t.Error("Eq should not match different string")
	}
	// Numeric stored as float64, compared as string via Sprint.
	if !Eq("age", float64(30)).Match(data) {
		t.Error("Eq should match equal float64")
	}
	if Eq("age", float64(31)).Match(data) {
		t.Error("Eq should not match different float64")
	}
}

func TestNe(t *testing.T) {
	data := map[string]any{"name": "alice"}

	if Ne("name", "bob").Match(data) == false {
		t.Error("Ne should match different value")
	}
	if Ne("name", "alice").Match(data) {
		t.Error("Ne should not match equal value")
	}
}

func TestGt(t *testing.T) {
	data := map[string]any{"score": float64(85)}

	if !Gt("score", 80).Match(data) {
		t.Error("Gt should match 85 > 80")
	}
	if Gt("score", 85).Match(data) {
		t.Error("Gt should not match 85 > 85")
	}
	if Gt("score", 90).Match(data) {
		t.Error("Gt should not match 85 > 90")
	}
}

func TestGte(t *testing.T) {
	data := map[string]any{"score": float64(85)}

	if !Gte("score", 85).Match(data) {
		t.Error("Gte should match 85 >= 85")
	}
	if !Gte("score", 80).Match(data) {
		t.Error("Gte should match 85 >= 80")
	}
	if Gte("score", 90).Match(data) {
		t.Error("Gte should not match 85 >= 90")
	}
}

func TestLt(t *testing.T) {
	data := map[string]any{"score": float64(85)}

	if !Lt("score", 90).Match(data) {
		t.Error("Lt should match 85 < 90")
	}
	if Lt("score", 85).Match(data) {
		t.Error("Lt should not match 85 < 85")
	}
	if Lt("score", 80).Match(data) {
		t.Error("Lt should not match 85 < 80")
	}
}

func TestLte(t *testing.T) {
	data := map[string]any{"score": float64(85)}

	if !Lte("score", 85).Match(data) {
		t.Error("Lte should match 85 <= 85")
	}
	if !Lte("score", 90).Match(data) {
		t.Error("Lte should match 85 <= 90")
	}
	if Lte("score", 80).Match(data) {
		t.Error("Lte should not match 85 <= 80")
	}
}

func TestIn(t *testing.T) {
	data := map[string]any{"lang": "go"}

	if !In("lang", "go", "python", "rust").Match(data) {
		t.Error("In should match when value in set")
	}
	if In("lang", "python", "rust").Match(data) {
		t.Error("In should not match when value not in set")
	}
}

func TestNin(t *testing.T) {
	data := map[string]any{"lang": "go"}

	if !Nin("lang", "python", "rust").Match(data) {
		t.Error("Nin should match when value not in set")
	}
	if Nin("lang", "go", "python").Match(data) {
		t.Error("Nin should not match when value in set")
	}
}

func TestContains(t *testing.T) {
	data := map[string]any{"desc": "hello world"}

	if !Contains("desc", "world").Match(data) {
		t.Error("Contains should match substring")
	}
	if Contains("desc", "xyz").Match(data) {
		t.Error("Contains should not match absent substring")
	}
	if !Contains("desc", "").Match(data) {
		t.Error("Contains should match empty substring")
	}
}

func TestNotContains(t *testing.T) {
	data := map[string]any{"desc": "hello world"}

	if !NotContains("desc", "xyz").Match(data) {
		t.Error("NotContains should match absent substring")
	}
	if NotContains("desc", "world").Match(data) {
		t.Error("NotContains should not match present substring")
	}
}

func TestAnd(t *testing.T) {
	data := map[string]any{"lang": "go", "score": float64(90)}

	f := And(Eq("lang", "go"), Gt("score", 80))
	if !f.Match(data) {
		t.Error("And should match when all sub-filters match")
	}

	f2 := And(Eq("lang", "go"), Gt("score", 95))
	if f2.Match(data) {
		t.Error("And should not match when one sub-filter fails")
	}

	// Empty And matches everything.
	if !And().Match(data) {
		t.Error("And with no filters should match")
	}
}

func TestOr(t *testing.T) {
	data := map[string]any{"lang": "go", "score": float64(90)}

	f := Or(Eq("lang", "python"), Gt("score", 80))
	if !f.Match(data) {
		t.Error("Or should match when at least one sub-filter matches")
	}

	f2 := Or(Eq("lang", "python"), Gt("score", 95))
	if f2.Match(data) {
		t.Error("Or should not match when no sub-filter matches")
	}

	// Empty Or matches nothing.
	if Or().Match(data) {
		t.Error("Or with no filters should not match")
	}
}

func TestFilterMissingField(t *testing.T) {
	data := map[string]any{"name": "alice"}

	// All comparisons on missing fields should return false.
	if Eq("missing", "x").Match(data) {
		t.Error("Eq on missing field should return false")
	}
	if Ne("missing", "x").Match(data) {
		t.Error("Ne on missing field should return false")
	}
	if Gt("missing", 0).Match(data) {
		t.Error("Gt on missing field should return false")
	}
	if Gte("missing", 0).Match(data) {
		t.Error("Gte on missing field should return false")
	}
	if Lt("missing", 0).Match(data) {
		t.Error("Lt on missing field should return false")
	}
	if Lte("missing", 0).Match(data) {
		t.Error("Lte on missing field should return false")
	}
	if In("missing", "x").Match(data) {
		t.Error("In on missing field should return false")
	}
	if Nin("missing", "x").Match(data) {
		t.Error("Nin on missing field should return false")
	}
	if Contains("missing", "x").Match(data) {
		t.Error("Contains on missing field should return false")
	}
	if NotContains("missing", "x").Match(data) {
		t.Error("NotContains on missing field should return false")
	}
}

func TestFilterNilData(t *testing.T) {
	// All filters on nil data should return false.
	if Eq("x", "y").Match(nil) {
		t.Error("Eq on nil data should return false")
	}
	if Ne("x", "y").Match(nil) {
		t.Error("Ne on nil data should return false")
	}
	if Gt("x", 0).Match(nil) {
		t.Error("Gt on nil data should return false")
	}
	if Gte("x", 0).Match(nil) {
		t.Error("Gte on nil data should return false")
	}
	if Lt("x", 0).Match(nil) {
		t.Error("Lt on nil data should return false")
	}
	if Lte("x", 0).Match(nil) {
		t.Error("Lte on nil data should return false")
	}
	if In("x", "y").Match(nil) {
		t.Error("In on nil data should return false")
	}
	if Nin("x", "y").Match(nil) {
		t.Error("Nin on nil data should return false")
	}
	if Contains("x", "y").Match(nil) {
		t.Error("Contains on nil data should return false")
	}
	if NotContains("x", "y").Match(nil) {
		t.Error("NotContains on nil data should return false")
	}
	if And(Eq("x", "y")).Match(nil) {
		t.Error("And on nil data should return false")
	}
	if Or(Eq("x", "y")).Match(nil) {
		t.Error("Or on nil data should return false")
	}
}

func TestNumericCoercion(t *testing.T) {
	// int stored as float64 (common after JSON unmarshal).
	data := map[string]any{"count": float64(42)}
	if !Gt("count", 41).Match(data) {
		t.Error("Gt should work with float64 field")
	}

	// int stored as int.
	data2 := map[string]any{"count": 42}
	if !Gt("count", 41).Match(data2) {
		t.Error("Gt should work with int field")
	}

	// int32
	data3 := map[string]any{"count": int32(42)}
	if !Gt("count", 41).Match(data3) {
		t.Error("Gt should work with int32 field")
	}

	// Non-numeric field should not match numeric comparison.
	data4 := map[string]any{"name": "alice"}
	if Gt("name", 0).Match(data4) {
		t.Error("Gt on non-numeric field should return false")
	}
}

func TestNestedFilters(t *testing.T) {
	data := map[string]any{
		"lang":  "go",
		"score": float64(85),
		"repo":  "tqdb",
	}

	// (lang == "go" AND score > 80) OR repo == "other"
	f := Or(
		And(Eq("lang", "go"), Gt("score", 80)),
		Eq("repo", "other"),
	)
	if !f.Match(data) {
		t.Error("nested filter should match")
	}

	// (lang == "python" AND score > 80) OR repo == "other"
	f2 := Or(
		And(Eq("lang", "python"), Gt("score", 80)),
		Eq("repo", "other"),
	)
	if f2.Match(data) {
		t.Error("nested filter should not match")
	}
}

func TestEqSpecialCharacters(t *testing.T) {
	data := map[string]any{"path": "/usr/bin/go", "emoji": "hello \U0001f30d"}

	if !Eq("path", "/usr/bin/go").Match(data) {
		t.Error("Eq should match path with slashes")
	}
	if !Eq("emoji", "hello \U0001f30d").Match(data) {
		t.Error("Eq should match emoji string")
	}
}

func TestEqBoolValues(t *testing.T) {
	data := map[string]any{"active": true, "deleted": false}

	if !Eq("active", true).Match(data) {
		t.Error("Eq should match true")
	}
	if Eq("active", false).Match(data) {
		t.Error("Eq should not match false when true")
	}
	if !Eq("deleted", false).Match(data) {
		t.Error("Eq should match false")
	}
}

func TestInEmptyValues(t *testing.T) {
	data := map[string]any{"lang": "go"}

	// Empty In set should never match.
	if In("lang").Match(data) {
		t.Error("In with no values should not match")
	}
}

func TestNinEmptyValues(t *testing.T) {
	data := map[string]any{"lang": "go"}

	// Empty Nin set should always match (value not in empty set).
	if !Nin("lang").Match(data) {
		t.Error("Nin with no values should match")
	}
}
