#include <metal_stdlib>

using namespace metal;

struct VertexOut {
  float4 position [[position]];
  half3 color;
  float3 edgeFlag;
};

VertexOut vertex vertexMain(uint vertexId [[vertex_id]],
                      device const float3* positions [[buffer(0)]],
                      device const float3* colors [[buffer(1)]]) {
  VertexOut o;
  o.position = float4(positions[vertexId], 1);
  o.color = half3(colors[vertexId]);
  if(vertexId % 3 == 0) {
    o.edgeFlag = {1, 0, 0};
  } else if (vertexId % 3 == 1) {
    o.edgeFlag = {0, 1, 0};
  } else {
    o.edgeFlag = {0, 0, 1};
  }
  return o;
}

half4 fragment fragmentMain(VertexOut in [[stage_in]]) {
  float a = min(min(in.edgeFlag[0], in.edgeFlag[1]), in.edgeFlag[2]);
  return a > 0.01 ? half4(in.color, 1.0) : half4(1, 0, 0, 1);
  // return half4(in.color, 1.0);
}
// vim:ft=cpp
