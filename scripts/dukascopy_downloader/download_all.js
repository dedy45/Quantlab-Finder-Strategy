/**
 * Download All Assets from Dukascopy
 * 
 * Batch download multiple symbols for multi-asset screening.
 * 
 * Usage:
 *   node download_all.js
 *   node download_all.js --years 5
 */

const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
    years: 10,
    timeframe: 'h1',
    output: '../../data/raw',
    
    // Assets to download
    symbols: [
        // Forex Majors
        'eurusd',
        'gbpusd',
        'usdjpy',
        'usdchf',
        'audusd',
        
        // Metals
        'xauusd',  // Gold
        'xagusd',  // Silver
        
        // Indices
        'spxusd',  // S&P 500
        'nsxusd',  // NASDAQ
        
        // Crypto (optional)
        // 'btcusd',
        // 'ethusd',
    ]
};

// Parse args
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
    if (args[i] === '--years' || args[i] === '-y') {
        CONFIG.years = parseInt(args[++i]);
    }
    if (args[i] === '--timeframe' || args[i] === '-t') {
        CONFIG.timeframe = args[++i];
    }
}

function toCSV(data, symbol) {
    const header = 'Symbol,DateTime,Open,High,Low,Close,Volume';
    const rows = data.map(bar => {
        const dt = new Date(bar.timestamp);
        const dateStr = dt.toISOString().replace('T', ' ').replace('Z', '').slice(0, 19);
        return `${symbol.toUpperCase()},${dateStr},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume || 0}`;
    });
    return [header, ...rows].join('\n');
}

async function downloadSymbol(symbol, startDate, endDate) {
    try {
        const data = await getHistoricalRates({
            instrument: symbol,
            dates: {
                from: startDate,
                to: endDate
            },
            timeframe: CONFIG.timeframe,
            priceType: 'bid',
            volumes: true
        });
        return data;
    } catch (error) {
        console.error(`  [FAIL] ${symbol}: ${error.message}`);
        return null;
    }
}

async function main() {
    console.log('============================================================');
    console.log('DUKASCOPY BATCH DOWNLOADER');
    console.log('============================================================');
    console.log(`Symbols: ${CONFIG.symbols.length}`);
    console.log(`Timeframe: ${CONFIG.timeframe}`);
    console.log(`Years: ${CONFIG.years}`);
    console.log('');

    const endDate = new Date();
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - CONFIG.years);

    console.log(`Period: ${startDate.toISOString().split('T')[0]} to ${endDate.toISOString().split('T')[0]}`);
    console.log('');

    // Ensure output directory
    const outputDir = path.resolve(__dirname, CONFIG.output);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const results = [];
    const totalStart = Date.now();

    for (let i = 0; i < CONFIG.symbols.length; i++) {
        const symbol = CONFIG.symbols[i];
        const progress = `[${i + 1}/${CONFIG.symbols.length}]`;
        
        process.stdout.write(`${progress} Downloading ${symbol.toUpperCase()}...`);
        
        const start = Date.now();
        const data = await downloadSymbol(symbol, startDate, endDate);
        const elapsed = ((Date.now() - start) / 1000).toFixed(1);

        if (data && data.length > 0) {
            // Save CSV
            const csv = toCSV(data, symbol);
            const filename = `${symbol.toUpperCase()}_dukascopy_${CONFIG.timeframe.toUpperCase()}.csv`;
            const filepath = path.join(outputDir, filename);
            fs.writeFileSync(filepath, csv);
            
            const fileSize = (fs.statSync(filepath).size / 1024 / 1024).toFixed(2);
            console.log(` [OK] ${data.length.toLocaleString()} bars, ${fileSize}MB, ${elapsed}s`);
            
            results.push({
                symbol: symbol.toUpperCase(),
                bars: data.length,
                file: filename,
                size: fileSize,
                status: 'OK'
            });
        } else {
            console.log(` [FAIL]`);
            results.push({
                symbol: symbol.toUpperCase(),
                bars: 0,
                file: null,
                size: 0,
                status: 'FAIL'
            });
        }

        // Small delay to avoid rate limiting
        await new Promise(r => setTimeout(r, 1000));
    }

    const totalElapsed = ((Date.now() - totalStart) / 1000).toFixed(1);

    // Summary
    console.log('');
    console.log('============================================================');
    console.log('DOWNLOAD SUMMARY');
    console.log('============================================================');
    
    const successful = results.filter(r => r.status === 'OK');
    const failed = results.filter(r => r.status === 'FAIL');
    
    console.log(`Total: ${results.length} symbols`);
    console.log(`Success: ${successful.length}`);
    console.log(`Failed: ${failed.length}`);
    console.log(`Time: ${totalElapsed}s`);
    console.log('');

    if (successful.length > 0) {
        console.log('Downloaded files:');
        successful.forEach(r => {
            console.log(`  [OK] ${r.symbol}: ${r.bars.toLocaleString()} bars (${r.size}MB)`);
        });
    }

    if (failed.length > 0) {
        console.log('');
        console.log('Failed symbols:');
        failed.forEach(r => {
            console.log(`  [FAIL] ${r.symbol}`);
        });
    }

    console.log('');
    console.log('Next step:');
    console.log('  cd ../../');
    console.log('  python scripts/convert_csv_to_parquet.py --all');
}

main().catch(console.error);
