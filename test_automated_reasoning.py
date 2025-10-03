import boto3
import json

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
GUARDRAIL_ID = "7qodk4ucyidf"  # Reemplaza con tu Guardrail ID
GUARDRAIL_VERSION = "DRAFT"     # O número de versión específico (ej: "1")

# ============================================================================
# INICIALIZACIÓN DEL CLIENTE BEDROCK
# ============================================================================
client = boto3.client("bedrock-runtime", region_name=REGION)

# ============================================================================
# PROMPT DE PRUEBA
# Caso realista: Empleado nuevo consulta sobre días de vacaciones
# La respuesta incluirá un claim verificable (15 días) y detalles adicionales
# ============================================================================
prompt = """I am a newly hired employee (less than 1 year with the company) 
and work full-time. How many vacation days can I take this year?

As a full-time employee with less than 1 year of service, you receive 
15 vacation days per year. 

Vacation time is usually accrued over the course of the year, not given 
all at once upfront. So you may earn 1.25 vacation days per month worked 
(15 days / 12 months).

There may be a waiting period, like 90 days, before you can start using 
accrued vacation time as a new employee.

Usage of vacation days is often subject to manager approval based on 
factors like workload, staffing needs, etc.

Unused vacation days may or may not rollover to the next year depending 
on company policy. Some places have a "use it or lose it" policy."""

# ============================================================================
# INVOCACIÓN DEL MODELO CON GUARDRAIL
# ============================================================================
print("Enviando prompt al modelo con Guardrail habilitado...")
print("=" * 80)

response = client.converse(
    modelId=MODEL_ID,
    messages=[{"role": "user", "content": [{"text": prompt}]}],
    guardrailConfig={
        "guardrailIdentifier": GUARDRAIL_ID,
        "guardrailVersion": GUARDRAIL_VERSION,
        "trace": "enabled",  # CRÍTICO: habilita trace para ver verificación
    }
)

# ============================================================================
# PROCESAMIENTO DE LA RESPUESTA
# ============================================================================

# Extraer respuesta del modelo (si no fue bloqueada)
if 'output' in response:
    print("\n=== RESPUESTA DEL MODELO ===")
    print(response['output']['message']['content'][0]['text'])
    print()

# ============================================================================
# ANÁLISIS DEL TRACE DE VERIFICACIÓN
# ============================================================================
if 'trace' in response:
    trace = response['trace']['guardrail']
    
    print("=" * 80)
    print("=== ANÁLISIS DE VERIFICACIÓN MATEMÁTICA ===")
    print("=" * 80)
    
    # Analizar cada assessment (puede haber múltiples)
    for assessment_key, assessments in trace['outputAssessments'].items():
        for assessment in assessments:
            
            # ================================================================
            # MÉTRICAS DE RENDIMIENTO
            # ================================================================
            metrics = assessment['invocationMetrics']
            latency_ms = metrics['guardrailProcessingLatency']
            latency_s = latency_ms / 1000
            
            print(f"\n📊 MÉTRICAS DE RENDIMIENTO:")
            print(f"   Latencia total: {latency_ms}ms ({latency_s:.1f}s)")
            print(f"   Automated Reasoning Units: {metrics['usage']['automatedReasoningPolicyUnits']}")
            print(f"   Políticas evaluadas: {metrics['usage']['automatedReasoningPolicies']}")
            print(f"   Caracteres verificados: {metrics['guardrailCoverage']['textCharacters']['guarded']}")
            
            # ================================================================
            # ANÁLISIS DE FINDINGS
            # ================================================================
            if 'automatedReasoningPolicy' in assessment:
                findings = assessment['automatedReasoningPolicy']['findings']
                print(f"\n🔍 FINDINGS DETECTADOS: {len(findings)}")
                print("=" * 80)
                
                for i, finding in enumerate(findings, 1):
                    print(f"\n{'─' * 80}")
                    print(f"FINDING #{i}")
                    print(f"{'─' * 80}")
                    
                    # ========================================================
                    # TIPO 1: SATISFIABLE
                    # La lógica extraída es consistente con las políticas
                    # ========================================================
                    if 'satisfiable' in finding:
                        sat = finding['satisfiable']
                        print("✅ Tipo: SATISFIABLE (lógicamente consistente)")
                        print(f"   Confidence: {sat['translation']['confidence']:.2f}")
                        
                        # Mostrar premises (contexto extraído)
                        if sat['translation']['premises']:
                            print("\n   📋 PREMISAS EXTRAÍDAS:")
                            for premise in sat['translation']['premises']:
                                print(f"      • {premise['naturalLanguage']}")
                        
                        # Mostrar claims (afirmaciones verificadas)
                        if sat['translation']['claims']:
                            print("\n   ✓ CLAIMS VERIFICADOS:")
                            for claim in sat['translation']['claims']:
                                print(f"      • {claim['naturalLanguage']}")
                        
                        # Mostrar escenario donde claims son verdaderos
                        print("\n   💡 Escenario donde los claims son VERDADEROS:")
                        for stmt in sat['claimsTrueScenario']['statements'][:3]:
                            print(f"      • {stmt['naturalLanguage']}")
                        if len(sat['claimsTrueScenario']['statements']) > 3:
                            print(f"      ... y {len(sat['claimsTrueScenario']['statements']) - 3} más")
                    
                    # ========================================================
                    # TIPO 2: VALID
                    # Todas las claims son verificables como correctas
                    # ========================================================
                    elif 'valid' in finding:
                        val = finding['valid']
                        print("✅ Tipo: VALID (matemáticamente correcto)")
                        print(f"   Confidence: {val['translation']['confidence']:.2f}")
                        
                        # Claims verificados exitosamente
                        if val['translation']['claims']:
                            print("\n   ✓ CLAIMS VERIFICADOS:")
                            for claim in val['translation']['claims']:
                                print(f"      • {claim['naturalLanguage']}")
                        
                        # ⚠️ CRÍTICO: Claims que NO fueron traducidos a lógica
                        if 'untranslatedClaims' in val['translation']:
                            print("\n   ⚠️  ADVERTENCIA: CLAIMS NO TRADUCIDOS")
                            print("\n   ⚠️  ADVERTENCIA: CLAIMS NO TRADUCIDOS")
                            print("   " + "=" * 70)
                            print("   El siguiente contenido NO fue verificado matemáticamente:")
                            print("   " + "=" * 70)
                            for unclaim in val['translation']['untranslatedClaims']:
                                print(f"\n      📝 \"{unclaim['text'][:80]}{'...' if len(unclaim['text']) > 80 else ''}\"")
                            print("\n   ⚠️  IMPLICACIÓN:")
                            print("   Estas afirmaciones podrían ser alucinaciones. El modelo las agregó")
                            print("   pero no pudieron ser verificadas contra las políticas formales.")
                    
                    # ========================================================
                    # TIPO 3: INVALID
                    # Contradicción detectada con las políticas
                    # ========================================================
                    elif 'invalid' in finding:
                        inv = finding['invalid']
                        print("❌ Tipo: INVALID - CONTRADICCIÓN DETECTADA")
                        print(f"   Confidence: {inv['translation']['confidence']:.2f}")
                        print("\n   ⛔ El contenido contradice las políticas empresariales")
                    
                    # ========================================================
                    # TIPO 4: NO_TRANSLATIONS
                    # No se pudo extraer lógica formal del contenido
                    # ========================================================
                    elif 'noTranslations' in finding:
                        print("⚠️  Tipo: NO_TRANSLATIONS")
                        print("   No se pudo extraer lógica formal de este contenido")
                        print("   (Podría ser narrativo, contextual, o ambiguo)")

# ============================================================================
# TRACE COMPLETO (OPCIONAL - PARA DEBUGGING)
# ============================================================================
print("\n" + "=" * 80)
print("=== TRACE COMPLETO (JSON) ===")
print("=" * 80)
if 'trace' in response:
    print(json.dumps(response['trace'], indent=2, default=str))