#include "Vtt_um_serialdev_sloth_stream.h"
#include "verilated.h"
struct Sim { VerilatedContext ctx; Vtt_um_serialdev_sloth_stream dut{&ctx}; };
extern "C" {
void* stream_create(){return new Sim;}
void stream_destroy(void* p){delete static_cast<Sim*>(p);}
void stream_pins(void* p,int data,int ctrl,int reset){auto& d=static_cast<Sim*>(p)->dut;d.ena=1;d.ui_in=data;d.uio_in=ctrl;d.rst_n=reset;d.eval();}
void stream_tick(void* p){auto& d=static_cast<Sim*>(p)->dut;d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();}
int stream_flags(void* p){return static_cast<Sim*>(p)->dut.uio_out;}
int stream_oe(void* p){return static_cast<Sim*>(p)->dut.uio_oe;}
int stream_data(void* p){return static_cast<Sim*>(p)->dut.uo_out;}
}
