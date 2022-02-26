use std::{io::{Write, Read}, path::PathBuf};
use openqasm::{self as oq, GenericError, Linearize, ProgramVisitor};
use clap::{Parser, ArgEnum};

const SVG_NS: &'static str = "http://www.w3.org/2000/svg";

struct SVGRenderer<'a> {
    root: minidom::Element,
    qubits: Vec<(f32, f32)>,
    bits: Vec<(f32, f32)>,
    cond_x: (f32, f32, f32),
    cond_bits: Vec<(usize, u64)>,
    font: rustybuzz::Face<'a>,
    font_size: f32,
    gutter: f32,
    spacing: f32,
    margin: f32
}

impl<'a> SVGRenderer<'a> {
    fn draw_quantum_wire(&mut self, a: (f32, f32), b: (f32, f32)) {
        self.root.append_child(minidom::Element::builder("line", SVG_NS)
            .attr("x1", a.0.to_string()).attr("y1", a.1.to_string())
            .attr("x2", b.0.to_string()).attr("y2", b.1.to_string())
            .build());
    }

    fn draw_classical_wire(&mut self, a: (f32, f32), b: (f32, f32), vertical: bool) {
        if vertical {
            self.draw_quantum_wire((a.0 - self.margin/2.0, a.1), (b.0 - self.margin/2.0, b.1));
            self.draw_quantum_wire((a.0 + self.margin/2.0, a.1), (b.0 + self.margin/2.0, b.1));
        } else {
            self.draw_quantum_wire((a.0, a.1 - self.margin/2.0), (b.0, b.1 - self.margin/2.0));
            self.draw_quantum_wire((a.0, a.1 + self.margin/2.0), (b.0, b.1 + self.margin/2.0));
        }
    }

    fn draw_polyline(&mut self, points: &[(f32, f32)]) {
        let points: String = points.iter().map(|(x, y)| format!("{},{} ", x, y)).collect();

        self.root.append_child(minidom::Element::builder("polyline", SVG_NS)
            .attr("points", points)
            .build());
    }

    fn bounding_box(&self, text: &str) -> f32 {
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(text);
        let buffer = rustybuzz::shape(&self.font, &[], buffer);
        let width: i32 = buffer.glyph_positions().iter().map(|g| g.x_advance).sum();
        let width = width as f32 / self.font.units_per_em() as f32;
        
        width
    }

    fn draw_text(&mut self, left: f32, top: f32, bottom: f32, name: &str, sub: &str, sup: bool, boxed: bool) -> f32 {
        let bbox = self.bounding_box(name);
        let bboxs = self.bounding_box(sub);
        let width = 2.0 * self.margin + bbox * self.font_size + bboxs * self.font_size * 0.60;

        if boxed {
            self.root.append_child(minidom::Element::builder("rect", SVG_NS)
                .attr("x", left.to_string()).attr("y", top.to_string())
                .attr("height", (bottom - top).to_string())
                .attr("width", width.to_string())
                .build());
        }

        let xheight = self.font_size * self.font.x_height().unwrap_or(0) as f32 / self.font.units_per_em() as f32;
        let mut text = minidom::Element::builder("text", SVG_NS)
            .attr("x", (left + self.margin).to_string())
            .attr("y", (top/2.0 + bottom/2.0 + xheight * 0.6).to_string())
            .attr("fill", "currentColor")
            .attr("stroke", "none")
            .attr("textLength", (width - 2.0 * self.margin).to_string())
            .append(name)
            .build();

        if !sub.is_empty() {
            text.append_child(minidom::Element::builder("tspan", SVG_NS)
                .attr("dy", (if sup { -0.4 } else { 0.15 } * self.font_size).to_string())
                .attr("font-size", (self.font_size * 0.60).to_string())
                .append(sub)
                .build());
        }

        self.root.append_child(text);

        width
    }

    fn draw_circle(&mut self, x: f32, y: f32, radius: f32, filled: bool) {
        self.root.append_child(minidom::Element::builder("circle", SVG_NS)
            .attr("cx", x.to_string()).attr("cy", y.to_string())
            .attr("r", radius.to_string())
            .attr("fill", if filled { "currentColor" } else { "none" })
            .build());
    }

    fn extend_wire(&mut self, reg: usize, x: f32) {
        let (cx, cy) = self.qubits[reg];
        let offset = self.margin + self.font_size/2.0;
        if cx < x {
            self.draw_quantum_wire((cx, cy + offset), (x, cy + offset));
        }
    }

    fn extend_wire_classical(&mut self, reg: usize, x: f32) {
        let (cx, cy) = self.bits[reg];
        let offset = self.margin + self.font_size/2.0;
        if cx < x {
            self.draw_classical_wire((cx, cy + offset), (x, cy + offset), false);
        }
    }

    fn prepare_multi(&mut self, qubits: &[usize]) -> (f32, f32, f32) {
        let min = qubits.iter().min().cloned().unwrap();
        let max = qubits.iter().max().cloned().unwrap();

        let x = (min..=max).map(|q| self.qubits[q].0)
            .fold(0.0, |a: f32, b: f32| a.max(b)) + self.spacing;

        let ymin= self.qubits[min].1;
        let ymax = self.qubits[max].1 + self.font_size + 2.0 * self.margin;

        for q in min..=max {
            self.extend_wire(q, x);
        }

        (x, ymin, ymax)
    }

    fn finish_multi(&mut self, qubits: &[usize], x: f32, skip: f32) {
        let min = qubits.iter().min().cloned().unwrap();
        let max = qubits.iter().max().cloned().unwrap();

        for q in min..=max {
            self.qubits[q].0 = x + skip;
        }
    }

    fn finish(&mut self) -> (f32, f32) {
        let wq = self.qubits.iter().cloned().map(|x| x.0).fold(0.0, |a: f32, b: f32| a.max(b));
        let wc = self.bits.iter().cloned().map(|x| x.0).fold(0.0, |a: f32, b: f32| a.max(b));
        let w = wq.max(wc) + self.spacing + 4.0*self.margin;
        let h = if self.bits.len() > 0 {
            self.bits[self.bits.len() - 1].1
        } else {
            self.qubits[self.qubits.len() - 1].1  
        } + self.font_size + 6.0*self.margin;

        for q in 0..self.qubits.len() {
            self.extend_wire(q, w - 4.0*self.margin);
        }

        for q in 0..self.bits.len() {
            self.extend_wire_classical(q, w - 4.0 * self.margin);
        }

        (w, h)
    }
}

impl<'a> oq::GateWriter for &mut SVGRenderer<'a> {
    type Error = std::convert::Infallible;

    fn initialize(&mut self, qubits: &[oq::Symbol], bits: &[oq::Symbol]) -> Result<(), Self::Error> {
        let height = self.font_size + 2.0 * self.margin;
        let pitch = height + self.gutter;
        let mut y = 0.0;
        let mut maxwidth = 0.0;

        for qubit in qubits.iter() {
            let (name, idx) = qubit.as_str().split_once("[").unwrap();
            let idx = idx.trim_end_matches("]");

            let width = self.draw_text(0.0, y, y + height, name, idx, false, false);
            self.qubits.push((0.0, y));
            y += pitch;
            if width > maxwidth {
                maxwidth = width;
            }
        }

        for bit in bits.iter() {
            let (name, idx) = bit.as_str().split_once("[").unwrap();
            let idx = idx.trim_end_matches("]");

            let width = self.draw_text(0.0, y, y + height, name, idx, false, false);
            self.bits.push((0.0, y));
            y += pitch;
            if width > maxwidth {
                maxwidth = width;
            }
        }

        for i in 0..self.qubits.len() {
            self.qubits[i].0 = maxwidth + self.margin;
        }

        for i in 0..self.bits.len() {
            self.bits[i].0 = maxwidth + self.margin;
        }

        Ok(())
    }

    fn write_u(&mut self, theta: oq::Value, phi: oq::Value, lambda: oq::Value, reg: usize) -> Result<(), Self::Error> {
        let (cx, cy) = self.qubits[reg];
        self.extend_wire(reg, cx + self.spacing);

        const ZERO: oq::Value = oq::Value::ZERO;
        const PI: oq::Value = oq::Value::PI;
        const PI_2: oq::Value = oq::Value::PI_2;
        let npi_2 = PI_2.checked_neg().unwrap();
        let pi_4 = PI_2.checked_div(oq::Value::int(2)).unwrap();
        let npi_4 = pi_4.checked_neg().unwrap();

        let (name, sub, sup): (&'static str, String, bool) = match (theta, phi, lambda) {
            v if v == (ZERO, ZERO, ZERO) => ("I", String::new(), false),
            v if v == (PI, ZERO, PI) => ("X", String::new(), false),
            v if v == (PI, PI_2, PI_2) => ("Y", String::new(), false),
            v if v == (ZERO, ZERO, PI) => ("Z", String::new(), false),
            v if v == (PI_2, ZERO, PI) => ("H", String::new(), false),
            v if v == (ZERO, ZERO, PI_2) => ("S", String::new(), false),
            v if v == (ZERO, ZERO, npi_2) => ("S", "†".to_string(), true),
            v if v == (ZERO, ZERO, pi_4) => ("T", String::new(), false),
            v if v == (ZERO, ZERO, npi_4) => ("T", "†".to_string(), true),
            v if v == (theta, npi_2, PI_2) => ("X", format!("{}", theta), false),
            v if v == (theta, ZERO, ZERO) => ("Y", format!("{}", theta), false),
            v if v == (ZERO, ZERO, lambda) => ("Z", format!("{}", lambda), false),
            (theta, phi, lambda) => ("U", format!("{}:{}:{}", theta, phi, lambda), false)
        };

        let width = self.draw_text(
            cx + self.spacing, cy, cy + self.font_size + 2.0*self.margin, 
            name, sub.as_str(), sup, true
        );

        self.qubits[reg].0 += self.spacing + width;

        Ok(())
    }

    fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error> {
        let (_, cy1) = self.qubits[copy];
        let (_, cy2) = self.qubits[xor];
        let offset = self.margin + self.font_size/2.0;

        let (x, _, _) = self.prepare_multi(&[copy, xor]);

        self.draw_circle(x, cy1 + offset, self.spacing / 8.0, true);
        self.draw_circle(x, cy2 + offset, self.spacing / 4.0, false);

        let extra = if cy2 < cy1 { -self.spacing/4.0 } else { self.spacing/4.0 };
        self.draw_quantum_wire((x, cy1 + offset), (x, cy2 + offset + extra));

        self.finish_multi(&[copy, xor], x, 0.0);

        Ok(())
    }

    fn write_barrier(&mut self, regs: &[usize]) -> Result<(), Self::Error> {
        let (x, ymin, ymax) = self.prepare_multi(regs);
        let offset = self.margin + self.font_size/2.0;

        let mut points = Vec::new();
        let p = ((ymax - ymin + 2.0*self.margin - 2.0*offset) / (2.0 * self.margin)) as usize;
        for k in 0..=p {
            points.push((x - self.margin/4.0, ymin + offset + 2.0 * self.margin * k as f32 - 2.0 * self.margin));
            points.push((x + self.margin/4.0, ymin + offset + 2.0 * self.margin * k as f32 - 1.0 * self.margin));
        }
        points.push((x - self.margin/2.0, ymin + offset + 2.0 * self.margin * p as f32));
        
        self.draw_polyline(&points);

        self.finish_multi(regs, x, 0.0);
        
        Ok(())
    }

    fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error> {
        let (cx, cy) = self.qubits[reg];
        let offset = self.margin + self.font_size / 2.0;
        self.extend_wire(reg, cx + self.spacing);

        self.draw_quantum_wire(
            (cx + self.spacing, cy + offset - 2.0 * self.margin),
            (cx + self.spacing, cy + offset + 2.0 * self.margin)
        );

        let width = self.draw_text(
            cx + 1.5 * self.spacing, cy, cy + 2.0*offset, 
            "0", "", false, true
        );
        
        self.qubits[reg].0 = cx + 1.5 * self.spacing + width;
        Ok(())
    }

    fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error> {
        let (_, cy1) = self.qubits[from];
        let (_, cy2) = self.bits[to];
        let offset = self.margin + self.font_size/2.0;

        let x = (from..self.qubits.len()).map(|q| self.qubits[q].0)
            .fold(0.0, |a: f32, b: f32| a.max(b)) + self.spacing;

        self.root.append_child(minidom::Element::builder("rect", SVG_NS)
            .attr("x", x.to_string()).attr("y", cy1.to_string())
            .attr("height", (2.0 * offset).to_string())
            .attr("width", self.spacing.to_string())
            .build());
        self.root.append_child(minidom::Element::builder("path", SVG_NS)
            .attr("d", format!("M {0} {1} A {3} {3} 0 0 1 {2} {1}",
                x + self.margin, cy1 + 1.25*offset, x + self.spacing - self.margin, offset/2.0
            )).build());
        self.draw_circle(x + self.spacing/2.0, cy1 + 1.25*offset, self.margin/3.0, true);
        self.draw_quantum_wire(
            (x + self.spacing/2.0, cy1 + 1.25*offset),
            (x + self.spacing/2.0 + 0.75*offset, cy1 + 0.5*offset)
        );

        for q in from..self.qubits.len() {
            if q == from {
                self.extend_wire(q, x);
            } else {
                self.extend_wire(q, x + self.spacing);
            }
            self.qubits[q].0 = x + self.spacing;
        }

        for b in 0..=to {
            self.extend_wire_classical(b, x + self.spacing/2.0);
            self.bits[b].0 = x + self.spacing/2.0;
        }

        self.draw_classical_wire(
            (x + self.spacing/2.0, cy1 + 2.0 * offset),
            (x + self.spacing/2.0, cy2 + offset - 2.5*self.margin),
            true
        );

        self.draw_polyline(&[
            (x + self.spacing / 2.0 - self.margin, cy2 + offset - 2.5*self.margin),
            (x + self.spacing / 2.0 + self.margin, cy2 + offset - 2.5*self.margin),
            (x + self.spacing / 2.0, cy2 + offset - 0.5*self.margin),
            (x + self.spacing / 2.0 - self.margin, cy2 + offset - 2.5*self.margin),
            (x + self.spacing / 2.0 + self.margin, cy2 + offset - 2.5*self.margin),
        ]);

        Ok(())
    }

    fn write_opaque(&mut self, name: &oq::Symbol, params: &[oq::Value], regs: &[usize]) -> Result<(), Self::Error> {
        if name.as_str() == "cx" && params.len() == 0 && regs.len() == 2 {
            self.write_cx(regs[0], regs[1])
        } else if name.as_str() == "ccx" && regs.len() == 3 && params.len() == 0 {
            let (c1, c2, xor) = (regs[0], regs[1], regs[2]);
            let (_, cy1) = self.qubits[c1];
            let (_, cy2) = self.qubits[c2];
            let (_, cy3) = self.qubits[xor];
            let offset = self.margin + self.font_size/2.0;

            let (x, _, _) = self.prepare_multi(&[c1, c2, xor]);

            self.draw_circle(x, cy1 + offset, self.spacing / 8.0, true);
            self.draw_circle(x, cy2 + offset, self.spacing / 8.0, true);
            self.draw_circle(x, cy3 + offset, self.spacing / 4.0, false);

            let extra = if cy3 < cy1 { -self.spacing/4.0 } else { self.spacing/4.0 };
            self.draw_quantum_wire((x, cy1 + offset), (x, cy3 + offset + extra));
            self.draw_quantum_wire((x, cy1 + offset), (x, cy2 + offset));

            self.finish_multi(&[c1, c2, xor], x, 0.0);

            Ok(())
        } else if name.as_str() == "cz" && regs.len() == 2 && params.len() == 0 {
            let (c1, c2) = (regs[0], regs[1]);
            let (_, cy1) = self.qubits[c1];
            let (_, cy2) = self.qubits[c2];
            let offset = self.margin + self.font_size/2.0;

            let (x, _, _) = self.prepare_multi(&[c1, c2]);

            self.draw_circle(x, cy1 + offset, self.spacing / 8.0, true);
            self.draw_circle(x, cy2 + offset, self.spacing / 8.0, true);
            self.draw_quantum_wire((x, cy1 + offset), (x, cy2 + offset));

            self.finish_multi(&[c1, c2], x, 0.0);

            Ok(())
        } else if (
            ((name.as_str() == "cy" || name.as_str() == "ch") && params.len() == 0) || 
            ((name.as_str() == "crz" || name.as_str() == "cu1") && params.len() == 1) || 
            (name.as_str() == "cu3" && params.len() == 3)
        ) && regs.len() == 2 {
            let (name, sub) = match name.as_str() {
                "cy" => ("Y", String::new()),
                "ch" => ("H", String::new()),
                "crz" => ("Z", format!("{}", params[0])),
                "cu1" => ("U", format!("{}", params[0])),
                "cu3" => ("U", format!("{}:{}:{}", params[0], params[1], params[2])),
                _ => unreachable!()
            };

            let (control, target) = (regs[0], regs[1]);

            let (_, cy1) = self.qubits[control];
            let (_, cy2) = self.qubits[target];
            let offset = self.margin + self.font_size/2.0;

            let (x, _, _) = self.prepare_multi(&[control, target]);

            let width = self.draw_text(x, cy2, cy2 + 2.0*offset, name, &sub, false, true);

            self.draw_circle(x + width/2.0, cy1 + offset, self.spacing / 8.0, true);

            if cy1 < cy2 {
                self.draw_quantum_wire((x + width/2.0, cy1 + offset), (x + width/2.0, cy2));
            } else {
                self.draw_quantum_wire((x + width/2.0, cy1 + offset), (x + width/2.0, cy2 + 2.0*offset));
            }
            
            for q in control.min(target)..=control.max(target) {
                if q != target {
                    self.extend_wire(q, x + width);
                }
            }

            self.finish_multi(&[control, target], x, width);

            Ok(())
        } else if regs.len() > 1 {
            let offset = self.margin + self.font_size/2.0;

            let name = if params.is_empty() {
                name.to_string()
            } else {
                format!("{}({})", name, params.iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(", ")
                )
            };

            let (x, ymin, ymax) = self.prepare_multi(regs);

            let width = self.draw_text(x, ymin, ymax, &name, "", false, true);

            let mut rmwidth = 0.0;
            for (i, r) in regs.iter().enumerate() {
                let (_, cy) = self.qubits[*r];
                let rwidth = self.draw_text(x + width, cy, cy + offset*2.0, &format!("{}", i), "", false, true);
                rmwidth = rwidth.max(rmwidth);
            }

            let min = regs.iter().min().cloned().unwrap();
            let max = regs.iter().max().cloned().unwrap();
            for i in (min + 1)..max {
                if !regs.contains(&i) {
                    self.qubits[i].0 = x + width;
                    self.extend_wire(i, x + width + rmwidth);
                }
            }
            
            self.finish_multi(regs, x, width + rmwidth);

            Ok(())
        } else {
            const ZERO: oq::Value = oq::Value::ZERO;
            const PI: oq::Value = oq::Value::PI;
            const PI_2: oq::Value = oq::Value::PI_2;
            let npi_2 = PI_2.checked_neg().unwrap();
            let pi_4 = PI_2.checked_div(oq::Value::int(2)).unwrap();
            let npi_4 = pi_4.checked_neg().unwrap();

            let reg = regs[0];
            let uparams = match (name.as_str(), params.len()) {
                ("id", 0) => Some([ZERO, ZERO, ZERO]),
                ("x", 0) => Some([PI, ZERO, PI]),
                ("y", 0) => Some([PI, PI_2, PI_2]),
                ("z", 0) => Some([ZERO, ZERO, PI]),
                ("h", 0) => Some([PI_2, ZERO, PI]),
                ("s", 0) => Some([ZERO, ZERO, PI_2]),
                ("sdg", 0) => Some([ZERO, ZERO, npi_2]),
                ("t", 0) => Some([ZERO, ZERO, pi_4]),
                ("tdg", 0) => Some([ZERO, ZERO, npi_4]),
                ("rx", 1) => Some([params[0], npi_2, PI_2]),
                ("ry", 1) => Some([params[0], ZERO, ZERO]),
                ("rz", 1) => Some([ZERO, ZERO, params[0]]),
                ("u1", 1) => Some([ZERO, ZERO, params[0]]),
                ("u2", 2) => Some([PI_2, params[0], params[1]]),
                ("u3", 3) => Some([params[0], params[1], params[2]]),
                _ => None
            };

            if let Some([theta, phi, lambda]) = uparams {
                self.write_u(theta, phi, lambda, reg)
            } else {
                let (cx, cy) = self.qubits[reg];
                self.extend_wire(reg, cx + self.spacing);

                let name = if params.is_empty() {
                    name.to_string()
                } else {
                    format!("{}({})", name, params.iter()
                        .map(|v| format!("{}", v))
                        .collect::<Vec<_>>()
                        .join(", ")
                    )
                };

                let width = self.draw_text(
                    cx + self.spacing, cy, cy + self.font_size + 2.0*self.margin, 
                    &name, "", false, true
                );
        
                self.qubits[reg].0 += self.spacing + width;
        
                Ok(())
            }
        }
    }

    fn start_conditional(&mut self, reg: usize, size: usize, val: u64) -> Result<(), Self::Error> {
        if self.cond_bits.is_empty() {
            let (x, ymin, ymax) = self.prepare_multi(&[0, self.qubits.len()-1]);
            self.cond_x = (x, ymin, ymax);

            for b in 0..self.bits.len() {
                self.extend_wire_classical(b, x + self.spacing/2.0);
                self.bits[b].0 = x + self.spacing/2.0;
            }
    
            self.finish_multi(&[0, self.qubits.len()-1], x, -self.spacing/2.0);
        }

        self.cond_bits.extend((0..size).map(|i| (reg + i, (val >> i) & 1)));

        Ok(())
    }

    fn end_conditional(&mut self) -> Result<(), Self::Error> {
        if self.cond_bits.is_empty() {
            return Ok(())
        }

        let (sx, _, ymaxo) = self.cond_x;
        let offset = self.margin + self.font_size / 2.0;

        for (bit, value) in self.cond_bits.clone() {
            let (_, cy) = self.bits[bit];
            self.draw_circle(sx + self.spacing / 2.0, cy + offset, self.spacing/8.0, true);
            self.draw_classical_wire(
                (sx + self.spacing/2.0, cy + offset),
                (sx + self.spacing/2.0, ymaxo + self.margin),
                true
            );

            self.draw_text(
                sx + self.spacing/2.0 + self.margin/2.0, 
                cy + offset + self.spacing/8.0 + self.margin/2.0,
                cy + offset + self.spacing/8.0 + self.font_size,
                "", &value.to_string(), true, false
            );
        }

        let (ex, ymin, ymax) = self.prepare_multi(&[0, self.qubits.len()-1]);
        self.finish_multi(&[0, self.qubits.len() - 1], ex, -self.spacing/2.0);

        self.root.append_child(minidom::Element::builder("rect", SVG_NS)
            .attr("x", sx.to_string()).attr("y", (ymin - self.margin).to_string())
            .attr("height", (ymax - ymin + 2.0*self.margin).to_string())
            .attr("width", (ex - sx - self.spacing/2.0).to_string())
            .build());

        self.cond_bits.clear();

        Ok(())
    }
}

fn draw_diagram(
    program: oq::Program, font: (&[u8], u32), font_name: String,
    policy: oq::translate::ExpansionPolicy,
    stroke_weight: f32, color: &str, spacing: f32, 
    margin: f32, gutter: f32, font_size: f32
) -> Result<(String, f32, f32), oq::Errors> {
    let font = rustybuzz::Face::from_slice(font.0, font.1).unwrap();

    let mut renderer = SVGRenderer {
        root: minidom::Element::builder("g", SVG_NS)
            .attr("stroke-width", stroke_weight.to_string())
            .attr("fill", "none")
            .attr("color", color)
            .attr("stroke", color)
            .attr("stroke-linejoin", "bevel")
            .attr("font-size", font_size.to_string())
            .attr("style", format!("font-family: {}, sans-serif;", font_name))
            .build(),
        qubits: Vec::new(),
        bits: Vec::new(),
        cond_x: (0.0, 0.0, 0.0),
        cond_bits: Vec::new(),
        font, font_size, margin, gutter, spacing
    };
    let mut linearize = Linearize::new(&mut renderer).with_policy(policy);
    linearize.visit_program(&program).to_errors()?;
    let (width, height) = renderer.finish();

    let document = minidom::Element::builder("svg", SVG_NS)
        .attr("width", width.to_string())
        .attr("height", height.to_string())
        .attr("viewBox", format!("{} {} {} {}", 
            -2.0*margin, -2.0*margin, width, height
        ))
        .append(renderer.root)
        .build();

    let mut output = Vec::new();
    document.write_to(&mut output).unwrap();
    Ok((String::from_utf8(output).unwrap(), width, height))
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// The OpenQASM file to convert.
    input: PathBuf,

    /// The file path for the output.
    /// 
    /// If this is not provided, then the output filename
    /// will be the same as the input but with an extension
    /// added depending on the output type.
    #[clap(long, short)]
    output: Option<PathBuf>,

    /// The type of output to generate.
    /// 
    /// If you specify `auto`, then it will be automatically detected
    /// based on the filename of the output. If no output filename
    /// is provided or the extension of the filename is not recognized,
    /// an SVG will be produced.
    #[clap(long, short, arg_enum, default_value_t = OutputType::Auto)]
    r#type: OutputType,

    /// The depth to expand gate definitions.
    /// 
    /// Set this to zero to prevent any gates from being expanded.
    #[clap(long, short, default_value_t = 100)]
    depth: usize,

    /// The policy for expanding gate definitions.
    /// 
    /// In addition to the depth value, this option can be used to
    /// control exactly which definitions are expanded:
    /// `this` means only expand definitions from the input file,
    /// `user` means expand everything except definitions from `qelib1.inc`,
    /// `all` means expand all definitions,
    /// `none` means never expand definitions.
    #[clap(short, long, arg_enum, default_value_t = Policy::User)]
    policy: Policy,

    /// The resolution multiplier for PNG rendering.
    /// 
    /// Increase this to get a higher resolution output.
    #[clap(short, long, default_value_t = 4.0)]
    scale: f32,

    /// The font to use for rendering.
    /// 
    /// By default, this will be Arial. The system default font
    /// will be used if the selected font isn't installed.
    /// 
    /// Be aware that the font will not be embedded
    /// in the output SVG so any viewer must also have this font
    /// installed to view it correctly.
    #[clap(short, long, default_value = "Arial")]
    font: String,

    /// The font-size to use for rendering.
    #[clap(long, default_value_t = 16.0)]
    font_size: f32,

    /// The spacing between gates.
    #[clap(long, default_value_t = 20.0)]
    spacing: f32,

    /// The margin seperating lines and text.
    #[clap(long, default_value_t = 2.5)]
    margin: f32,

    /// The gutter to leave between rows of gates.
    #[clap(long, default_value_t = 10.0)]
    gutter: f32,

    /// The stroke weight to use for gates and wires.
    #[clap(long, default_value_t = 1.0)]
    stroke_weight: f32,

    /// The color to use for text and lines.
    /// 
    /// This should be a valid SVG color, for example "#cc0000" or "red".
    #[clap(short, long, default_value = "black")]
    color: String,

    /// The background color for PNG outputs.
    /// 
    /// This should be either a valid SVG color or "transparent" to
    /// specify a transparent background.
    /// 
    /// Note that a solid background is never generated for SVG outputs.
    #[clap(short, long, default_value = "white")]
    background: String
}

#[derive(Debug, Clone, ArgEnum)]
enum OutputType {
    PNG,
    SVG,
    Auto
}

#[derive(Debug, Clone, ArgEnum)]
enum Policy {
    This,
    User,
    All,
    None
}

fn main() {
    let args = Args::parse();

    let mut cache = oq::SourceCache::new();
    let mut parser = oq::Parser::new(&mut cache);
    let this_id = parser.parse_file(&args.input);

    let program = match parser.done().to_errors() {
        Ok(program) => program,
        Err(e) => {
            e.eprint(&mut cache).unwrap();
            std::process::exit(exitcode::DATAERR)
        }
    };

    if let Err(e) = program.type_check().to_errors() {
        e.eprint(&mut cache).unwrap();
        std::process::exit(exitcode::DATAERR)
    }

    let mut fontdb = usvg::fontdb::Database::new();
    fontdb.load_system_fonts();
    let font_id = fontdb.query(&usvg::fontdb::Query {
        families: &[
            usvg::fontdb::Family::Name(&args.font),
            usvg::fontdb::Family::SansSerif,
            usvg::fontdb::Family::Serif
        ],
        weight: usvg::fontdb::Weight::NORMAL,
        stretch: usvg::fontdb::Stretch::Normal,
        style: usvg::fontdb::Style::Normal
    }).unwrap();
    let face = fontdb.face(font_id).unwrap();
    let (source, index) = fontdb.face_source(font_id).unwrap();
    
    let font_data: Vec<u8> = match source {
        usvg::fontdb::Source::Binary(data) => (&*data).as_ref().to_vec(),
        usvg::fontdb::Source::SharedFile(_, data) => (&*data).as_ref().to_vec(),
        usvg::fontdb::Source::File(path) => {
            let mut buf = Vec::new();
            std::fs::File::open(path).unwrap().read_to_end(&mut buf).unwrap();
            buf
        }
    };
    

    let policy = match args.policy {
        Policy::All => oq::translate::ExpansionPolicy::new().depth(args.depth),
        Policy::This => oq::translate::ExpansionPolicy::new()
            .allow_file(this_id).depth(args.depth),
        Policy::User => match cache.get_id("qelib1.inc") {
            None => oq::translate::ExpansionPolicy::new(),
            Some(qelib1_id) => oq::translate::ExpansionPolicy::new()
                .deny_file(qelib1_id).depth(args.depth)
        },
        Policy::None => oq::translate::ExpansionPolicy::new().depth(0)
    };

    let (svg_data, width, height) = match draw_diagram(
        program, (&font_data, index), face.family.clone(), policy, 
        args.stroke_weight, &args.color, args.spacing, 
        args.margin, args.gutter, args.font_size
    ) {
        Ok(svg) => svg,
        Err(e) => {
            e.eprint(&mut cache).unwrap();
            std::process::exit(exitcode::DATAERR)
        }
    };

    let actual_type = match args.r#type {
        OutputType::Auto => {
            if let Some(output) = &args.output {
                match output.extension().map(|d| d.to_str()).flatten() {
                    Some("png") => OutputType::PNG,
                    Some("svg") => OutputType::SVG,
                    _ => OutputType::SVG
                }
            } else {
                OutputType::SVG
            }
        },
        x => x
    };

    let output = match args.output {
        Some(buf) => buf,
        None => {
            let mut output = args.input.clone();
            output.set_extension(match actual_type {
                OutputType::PNG => "png",
                OutputType::SVG => "svg",
                _ => unreachable!()
            });
            output
        }
    };

    match actual_type {
        OutputType::SVG => {
            std::fs::File::create(&output)
                .unwrap()
                .write_all(svg_data.as_bytes())
                .unwrap()
        },
        OutputType::PNG => {
            use std::str::FromStr;

            let bg = svgtypes::Color::from_str(&args.background).unwrap();
            let bg = tiny_skia::Color::from_rgba8(bg.red, bg.green, bg.blue, bg.alpha);
            let mut pixmap = tiny_skia::Pixmap::new(
                 (args.scale * width.ceil()) as u32, (args.scale * height.ceil()) as u32
            ).unwrap();
            pixmap.fill(bg);

            let mut opts = usvg::Options::default();
            opts.fontdb = fontdb;
            let tree = usvg::Tree::from_str(&svg_data, &opts.to_ref()).unwrap();
            resvg::render(
                &tree, 
                usvg::FitTo::Zoom(args.scale), 
                tiny_skia::Transform::default(), 
                pixmap.as_mut()
            ).unwrap();

            let png_data = pixmap.encode_png().unwrap();
            std::fs::File::create(&output)
                .unwrap()
                .write_all(&png_data)
                .unwrap();
        },
        _ => unreachable!()
    }
}