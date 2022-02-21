use openqasm as oq;
use oq::{GenericError, ProgramVisitor};
use plotters::{prelude::*, coord::types::RangedCoordf32, style::RGBAColor};

struct SVGRenderer<'a> {
    backend: DrawingArea<SVGBackend<'a>, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    qubits: Vec<f32>,
    bits: Vec<f32>,
    style: ShapeStyle
}

impl<'a> SVGRenderer<'a> {
    fn reg_loc(&self, reg: usize) -> (f32, f32) {
        (self.qubits[reg], 20.0 * reg as f32 + 10.0)
    }

    fn extend_wire(&mut self, reg: usize, x: f32) {
        let (cx, cy) = self.reg_loc(reg);
        if cx < x {
            self.backend.draw(&PathElement::new([
                (cx, cy),
                (x, cy),
            ], self.style.clone())).unwrap();
        }
    }

    fn prepare_multi(&mut self, qubits: &[usize], spacing: f32, skip: f32) -> (f32, f32, f32) {
        let min = qubits.iter().min().cloned().unwrap();
        let max = qubits.iter().max().cloned().unwrap();

        let x = (min..=max).map(|q| self.qubits[q])
            .fold(0.0, |a: f32, b: f32| a.max(b)) + spacing;

        let (_, ymin) = self.reg_loc(min);
        let (_, ymax) = self.reg_loc(max);

        for q in min..=max {
            self.extend_wire(q, x);
            self.qubits[q] = x + skip;
        }

        (x, ymin, ymax)
    }

    fn finish(&mut self) -> (f32, f32) {
        let wq = self.qubits.iter().cloned().fold(0.0, |a: f32, b: f32| a.max(b));
        let wc = self.bits.iter().cloned().fold(0.0, |a: f32, b: f32| a.max(b));
        let w = wq.max(wc) + 10.0;

        let h = 20.0 * (self.qubits.len() + self.bits.len() + 1) as f32;

        for q in 0..self.qubits.len() {
            self.extend_wire(q, w);
        }

        (w, h)
    }
}

impl<'a> oq::GateWriter for &mut SVGRenderer<'a> {
    type Error = std::convert::Infallible;

    fn initialize(&mut self, qubits: &[oq::Symbol], bits: &[oq::Symbol]) -> Result<(), Self::Error> {
        for (i, qubit) in qubits.iter().enumerate() {
            self.backend.draw(&Text::new(
                qubit.as_str(), (0.0, 20.0 * i as f32), ("monospace", 20.0).into_font()
            )).unwrap();
            self.qubits.push(50.0);
        }

        for (i, bit) in bits.iter().enumerate() {
            self.backend.draw(&Text::new(
                bit.as_str(), (0.0, 20.0 * (i + self.qubits.len()) as f32), ("monospace", 20.0).into_font()
            )).unwrap();
            self.bits.push(50.0);
        }

        Ok(())
    }

    fn write_u(&mut self, theta: oq::Value, phi: oq::Value, lambda: oq::Value, reg: usize) -> Result<(), Self::Error> {
        let (cx, cy) = self.reg_loc(reg);

        self.extend_wire(reg, cx + 10.0);
        self.backend.draw(&Rectangle::new([
            (cx + 10.0, cy - 8.0), (cx + 30.0, cy + 8.0)
        ], self.style.clone())).unwrap();

        const ZERO: oq::Value = oq::Value::ZERO;
        const PI: oq::Value = oq::Value::PI;
        const PI_2: oq::Value = oq::Value::PI_2;
        let npi_2 = PI_2.checked_neg().unwrap();
        let pi_4 = PI_2.checked_div(oq::Value::int(2)).unwrap();
        let npi_4 = pi_4.checked_neg().unwrap();

        let name: std::borrow::Cow<'static, str> = match (theta, phi, lambda) {
            v if v == (ZERO, ZERO, ZERO) => "I".into(),
            v if v == (PI, ZERO, PI) => "X".into(),
            v if v == (PI, PI_2, PI_2) => "Y".into(),
            v if v == (ZERO, ZERO, PI) => "Z".into(),
            v if v == (PI_2, ZERO, PI) => "H".into(),
            v if v == (ZERO, ZERO, PI_2) => "S".into(),
            v if v == (ZERO, ZERO, npi_2) => "S*".into(),
            v if v == (ZERO, ZERO, pi_4) => "T".into(),
            v if v == (ZERO, ZERO, npi_4) => "T*".into(),
            v if v == (theta, npi_2, PI_2) => format!("RX({})", theta).into(),
            v if v == (theta, ZERO, ZERO) => format!("RY({})", theta).into(),
            v if v == (ZERO, ZERO, lambda) => format!("RZ({})", lambda).into(),
            (theta, phi, lambda) => format!("U({},{},{})", theta, phi, lambda).into()
        };
        let name = name.as_ref();

        self.backend.draw(&Text::new(
            name, (cx + 17.5 - 2.5 * name.len()  as f32, cy - 7.0), ("monospace", 20.0).into_font()
        )).unwrap();

        self.qubits[reg] += 30.0;

        Ok(())
    }

    fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error> {
        let (_, cy1) = self.reg_loc(copy);
        let (_, cy2) = self.reg_loc(xor);

        let (x, _, _) = self.prepare_multi(&[copy, xor], 10.0, 0.0);

        self.backend.draw(&Circle::new((x, cy1), 2.5f32, self.style.clone().filled())).unwrap();
        self.backend.draw(&Circle::new((x, cy2), 5.0f32, self.style.clone())).unwrap();
        self.backend.draw(&PathElement::new([
            (x, cy1), (x, cy2 + if cy2 < cy1 { -5.0 } else { 5.0 })
        ], self.style.clone())).unwrap();

        Ok(())
    }

    fn write_barrier(&mut self, regs: &[usize]) -> Result<(), Self::Error> {
        let (x, ymin, ymax) = self.prepare_multi(regs, 10.0, 0.0);

        let mut points = Vec::new();
        let p = ((ymax - ymin + 10.0) / 5.0) as usize;
        for k in 0..=p {
            points.push((x - 1.0, ymin + 5.0 * k as f32 - 5.0));
            points.push((x + 1.0, ymin + 5.0 * k as f32 - 2.5));
        }

        self.backend.draw(&PathElement::new(points, self.style.clone())).unwrap();

        Ok(())
    }

    fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error> {
        let (cx, cy) = self.reg_loc(reg);
        self.extend_wire(reg, cx + 10.0);
        
        self.backend.draw(&PathElement::new([
            (cx + 10.0, cy - 5.0), (cx + 10.0, cy + 5.0)
        ], self.style.clone())).unwrap();

        self.backend.draw(&PathElement::new([
            (cx + 15.0, cy - 7.0), (cx + 15.0, cy + 7.0)
        ], self.style.clone())).unwrap();

        self.backend.draw(&PathElement::new([
            (cx + 25.0, cy - 7.0), (cx + 27.0, cy), (cx + 25.0, cy + 7.0)
        ], self.style.clone())).unwrap();
        
        self.backend.draw(&Text::new(
            "0", (cx + 16.0, cy - 7.0), ("monospace", 20.0).into_font()
        )).unwrap();

        self.qubits[reg] = cx + 30.0;

        Ok(())
    }

    fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_opaque(&mut self, name: &oq::Symbol, params: &[oq::Value], regs: &[usize]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn start_conditional(&mut self, reg: usize, size: usize, val: u64) -> Result<(), Self::Error> {
        Ok(())
    }

    fn end_conditional(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn make_diagram(path: &str, cache: &mut oq::SourceCache) -> Result<String, oq::Errors> {
    let mut parser = oq::Parser::new(cache);
    parser.parse_file(path);
    let program = parser.done().to_errors()?;
    program.type_check().to_errors()?;

    let mut output = String::new();
    let (width, height) = {
        let backend = SVGBackend::with_string(&mut output, (i32::MAX as u32, i32::MAX as u32));
        let area = backend.into_drawing_area()
            .apply_coord_spec(Cartesian2d::<RangedCoordf32, RangedCoordf32>::new(
                0.0f32..i32::MAX as f32,
                0.0f32..i32::MAX as f32,
                (0..i32::MAX, 0..i32::MAX),
            ));
        let mut renderer = SVGRenderer {
            backend: area,
            qubits: Vec::new(),
            bits: Vec::new(),
            style: ShapeStyle {
                color: BLACK.to_rgba(),
                filled: false,
                stroke_width: 1
            }
        };
        let mut l = oq::translate::Linearize::new(&mut renderer, usize::MAX);
        l.visit_program(&program).to_errors()?;
        renderer.finish()
    };
    let (width, height) = (width.ceil() as u32, height.ceil() as u32);

    output = output.replace(
        r#"<svg width="2147483647" height="2147483647" viewBox="0 0 2147483647 2147483647" xmlns="http://www.w3.org/2000/svg">"#,
        &format!(r#"<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">"#)
    );

    Ok(output)
}

fn main() {
    let filename = std::env::args().skip(1).next().map(String::from).unwrap_or_default();
    let mut cache = oq::SourceCache::new();
    let res = make_diagram(&filename, &mut cache);
    match res {
        Ok(string) => println!("{}", string),
        Err(err) => err.print(&mut cache).unwrap()
    }
}
