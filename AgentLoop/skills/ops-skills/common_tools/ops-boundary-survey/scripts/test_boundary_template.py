"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
边界测试框架 - <算子名称>
"""
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving


class Test<OpName>Boundary(TestCase):
    """<算子名称> 边界测试套件。

    测试方向：<仅正向 / 仅反向 / 双向>（按用户要求填写）
    每个边界 case 只生成用户要求方向的 test 方法。
    """

    # ===== 输入构造（通用，正反向复用）=====

    @golden_data_cache(__file__)
    def gen_input_data(self, shape1, shape2, attr1, attr2):
        """生成测试输入数据。"""
        input1 = torch.rand(shape1)
        input2 = torch.rand(shape2)
        return input1, input2

    # ===== 正向（Forward）：只验证 output =====

    def golden_forward(self, input1, input2, attr1, attr2):
        """Golden 前向实现，返回 output。"""
        # 例：
        # input1_cpu = input1.clone()
        # input2_cpu = input2.clone()
        # return <golden_impl>(input1_cpu, input2_cpu, attr1=attr1, attr2=attr2)
        pass

    def npu_forward(self, input1, input2, attr1, attr2):
        """NPU 前向实现，返回 output。"""
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        return mx_driving.<op_name>(input1_npu, input2_npu, attr1=attr1, attr2=attr2)

    def test_<case_name>_forward(self):
        """约束内用例（正向）：只对比 output。"""
        input1, input2 = self.gen_input_data(shape1, shape2, attr1, attr2)
        golden_out = self.golden_forward(input1, input2, attr1, attr2)
        npu_out = self.npu_forward(input1, input2, attr1, attr2)
        self.assertRtolEqual(npu_out, golden_out.detach().cpu())

    # ===== 反向（Backward）：只验证梯度 =====

    def golden_backward(self, input1, input2, attr1, attr2, grad_output):
        """Golden 反向：CPU/golden 上前向+反向，返回各输入梯度。"""
        # 例：
        # input1_cpu = input1.clone().requires_grad_(True)
        # input2_cpu = input2.clone().requires_grad_(True)
        # output = <golden_impl>(input1_cpu, input2_cpu, attr1=attr1, attr2=attr2)
        # output.backward(grad_output, retain_graph=True)
        # return input1_cpu.grad, input2_cpu.grad
        pass

    def npu_backward(self, input1, input2, attr1, attr2, grad_output):
        """NPU 反向：NPU 上前向+反向，返回各输入梯度。"""
        input1_npu = input1.detach().clone().npu().requires_grad_(True)
        input2_npu = input2.detach().clone().npu().requires_grad_(True)
        output = mx_driving.<op_name>(input1_npu, input2_npu, attr1=attr1, attr2=attr2)
        output.backward(grad_output.npu(), retain_graph=True)
        return input1_npu.grad.cpu(), input2_npu.grad.cpu()

    def test_<case_name>_backward(self):
        """约束内用例（反向）：只对比梯度。"""
        input1, input2 = self.gen_input_data(shape1, shape2, attr1, attr2)
        grad_output = torch.rand_like(
            self.npu_forward(input1, input2, attr1, attr2).cpu().detach()
        )
        golden_grads = self.golden_backward(input1, input2, attr1, attr2, grad_output)
        npu_grads = self.npu_backward(input1, input2, attr1, attr2, grad_output)
        for npu_g, golden_g in zip(npu_grads, golden_grads):
            self.assertRtolEqual(npu_g, golden_g)

    # ===== 约束外测试（异常拦截，按方向覆盖）=====

    def test_<error_case_name>_error(self):
        """约束外：验证异常正确拦截。"""
        input1, input2 = self.gen_input_data(shape1, shape2, attr1, attr2)

        with self.assertRaises(Exception) as ctx:
            mx_driving.<op_name>(input1.npu(), input2.npu(), attr1=<invalid_value>, ...)

        self.assertEqual(str(ctx.exception), "<预期错误消息>\n")


if __name__ == "__main__":
    torch.npu.conv.allow_hf32 = False
    run_tests()