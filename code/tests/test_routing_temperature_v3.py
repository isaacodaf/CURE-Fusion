"""Algebra and autograd tests; these do not replace actual CUDA admission."""
import unittest
import numpy as np
import torch
from cure_fusion.routing_temperature_v3 import fit_routing_temperatures, scaled_routing_bias


class RoutingTemperatureTests(unittest.TestCase):
    def test_masked_rms_and_unchanged_small_modality(self):
        values=np.array([[[3.,.01],[4.,-.02],[np.nan,np.nan]]])
        mask=np.array([[True,True,False]])
        temperature,rms=fit_routing_temperatures(values,mask)
        np.testing.assert_allclose(rms,[np.sqrt(12.5),np.sqrt(.00025)])
        self.assertEqual(temperature[1],.75)

    def test_zero_and_signed_order_preserved(self):
        values=torch.tensor([[-100.,-1.],[0.,0.],[100.,1.]],dtype=torch.float64)
        bias=scaled_routing_bias(values,torch.tensor([11.,.75],dtype=torch.float64))
        self.assertTrue(torch.equal(torch.sign(values),torch.sign(bias)))
        self.assertTrue(bool((bias.abs()<=.5).all()))
        self.assertTrue(bool((bias[1:]>bias[:-1]).all()))

    def test_camera_gradient_restored_without_radar_change(self):
        values=torch.tensor([[10.,.03]],dtype=torch.float64,requires_grad=True)
        temperature=torch.tensor([11.671275371025082,.75],dtype=torch.float64)
        new=scaled_routing_bias(values,temperature)
        old=.5*torch.tanh(values/.75)
        self.assertTrue(torch.equal(new[:,1],old[:,1]))
        new.sum().backward()
        self.assertGreater(float(values.grad[0,0]),.001)

    def test_original_temperature_is_exact_identity(self):
        values=torch.linspace(-20,20,60).reshape(30,2)
        actual=scaled_routing_bias(values,torch.tensor([.75,.75]))
        self.assertTrue(torch.equal(actual,.5*torch.tanh(values/.75)))

    def test_invalid_targets_rejected(self):
        for mean,mask in [(np.full((1,2),np.nan),np.array([True])),
                          (np.zeros((1,2)),np.array([False]))]:
            with self.assertRaises(ValueError):fit_routing_temperatures(mean,mask)

    def test_no_input_mutation(self):
        value=torch.tensor([[8.,.03]],requires_grad=True)
        before=value.detach().clone()
        scaled_routing_bias(value,torch.tensor([11.,.75])).sum().backward()
        self.assertTrue(torch.equal(value.detach(),before))


if __name__=='__main__':unittest.main()
