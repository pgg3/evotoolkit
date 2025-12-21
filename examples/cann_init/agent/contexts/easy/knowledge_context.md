## API Reference

### Relu
- **Signature**: `(1) void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
(2) void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
(3) void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)`
- **Description**: dst[i] = (src[i] < 0) ? 0 : src[i]

### DataCopy
- **Signature**: `(1) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const Nd2NzParams& intriParams)
(2) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcGlobal, const Nd2NzParams& intriParams)
(3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const DataCopyParams& repeatParams)
(4) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const DataCopyParams& repeatParams)
(5) void DataCopy(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src_T>& srcLocal, const DataCopyParams& repeatParams)
... (共 17 个重载)`
- **Description**: format transform(such as nd2nz) during data load from OUT to L1

### GetBlockIdx
- **Signature**: `int64_t GetBlockIdx()`

## Example Reference

### squared_relu
**Why Selected**: This operator uses Relu as part of its computation pipeline, demonstrating the exact API usage, multi-core tiling, ping-pong buffering, and data type handling patterns needed for implementing Relu.

**Adaptation Notes**: - Remove the Mul operation (keep only Relu)
- Simplify Compute to only call Relu without squaring
- Keep the same multi-core distribution logic, ping-pong buffer management, and Cast operations for different data types
- Maintain the same DataCopyPad and event synchronization patterns

**Init Pattern** (buffer setup):
```cpp
__aicore__ inline void SquaredReluND<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                               const SquaredReluTilingData* tilingData) {
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;
    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}
```

**Process Pattern** (main loop):
```cpp
__aicore__ inline void SquaredReluND<T>::Process() {
    if (blockIdx >= needCoreNumber) {
        return;
    }
    int64_t totalTimes = elementNum / PP_ELEMENT_NUM;
    int64_t remain = elementNum % PP_ELEMENT_NUM;
    if (remain > 0) {
        totalTimes++;
    }
    int64_t loopNum = totalTimes / needCoreNumber;
    int64_t loopRemain = totalTimes % needCoreNumber;
    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    int64_t eachCoreStartOffset = loopNum * blockIdx * PP_ELEMENT_NUM;
    if (loopRemain > 0) {
        if (blockIdx >= loopRemain) {
            eachCoreStartOffset += elementNum % (PP_ELEMENT_NUM * needCoreNumber);
        }
    }
    int32_t calNum = PP_ELEMENT_NUM;
    int64_t lastCoreNum = loopRemain == 0 ? needCoreNumber - 1 : loopRemain - 1;
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t localOffset = i * PP_ELEMENT_NUM;
        if (remain > 0 && i == loopNum -1 && blockIdx == lastCoreNum) {
            calNum = remain;
        }
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        CopyInAndCast(eachCoreStartOffset + localOffset, calNum);
        Compute(calNum);
        CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}
```

**Compute Pattern** (core computation):
```cpp
__aicore__ inline void SquaredReluND<T>::Compute(int64_t dataCount) {
    Relu(xTensorFp32, xTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
    Mul(xTensorFp32, xTensorFp32, xTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}
```

**CopyIn/CopyOut Pattern** (data movement):
```cpp
// CopyInAndCast
__aicore__ inline void SquaredReluND<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount) {
    xTensor = pingPongFlag ?
            tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() :
            tmpTensor[0].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int32_t elementByte = PP_ELEMENT_NUM * sizeof(T);
        xTmp = pingPongFlag ?
                tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte].ReinterpretCast<T>();
        DataCopyPad(xTmp, inputGm[inputOffset], dataCopyParams, padParams);
    } else {
        DataCopyPad(xTensor, inputGm[inputOffset], dataCopyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    xTensorFp32 = xTensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(xTensorFp32, xTmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

// CastAndCopyOut
__aicore__ inline void SquaredReluND<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount) {
    if (std::is_same_v<T, half>) {
        Cast(xTensor, xTensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(xTensor, xTensorFp32, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outputOffset], xTensor, dataCopyParams);
}
```

**Key API Calls** (use these exact patterns):
```cpp
inputGm.SetGlobalBuffer((__gm__ T*)input);
outputGm.SetGlobalBuffer((__gm__ T*)output);
pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
tmpTensor = ubTBuf.Get<uint8_t>();
GetBlockIdx();
xTensor = tmpTensor[offset].ReinterpretCast<T>();
xTensorFp32 = xTensor.template ReinterpretCast<float>();
DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
DataCopyPad(dst, src, dataCopyParams, padParams);
DataCopyPad(dst, src, dataCopyParams);
Cast(xTensorFp32, xTmp, RoundMode::CAST_NONE, dataCount);
Cast(xTensor, xTensorFp32, RoundMode::CAST_RINT, dataCount);
Relu(xTensorFp32, xTensorFp32, dataCount);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE2_V>(eventId);
SetFlag<HardEvent::V_MTE3>(eventId);
WaitFlag<HardEvent::MTE3_MTE2>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::V_MTE3>(eventId);
```

